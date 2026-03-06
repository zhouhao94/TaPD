from typing import List
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

class Av2Dataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str = None,
        candidate_times: List[int] = [0, 10, 20, 30, 40, 50],
        radius: float = 150.0,
        train_mode: str = 'only_focal',
        val_sequence_start: int = 0,
        preload_to_memory: bool = False,  # 新增参数，是否预加载到内存
    ):
        assert train_mode in ['only_focal', 'focal_and_scored']
        assert split in ['train', 'val', 'test']
        super(Av2Dataset, self).__init__()                      
        self.split = split
        self.data_folder = Path(data_root) / split
        self.file_list = sorted(list(self.data_folder.glob('*.pt')))
        self.preload_to_memory = preload_to_memory
        self.data_list = None
        if self.preload_to_memory:
            print(f"Preloading {len(self.file_list)} files into memory...")
            self.data_list = [torch.load(f) for f in self.file_list]
            print("Preload finished.")
        
        """
        小数据集测试代码
        """
        # if split == 'train':
        #     self.file_list = self.file_list[::30]

        self.num_future_steps = 0 if split =='test' else 60
        self.candidate_times = candidate_times
        self.mode = 'only_focal' if split != 'train' else train_mode
        self.radius = radius
        self.val_sequence_start = val_sequence_start
        print(
            f'data root: {data_root}/{split}, total number of files: {len(self.file_list)}'
        )

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: int):
        if self.preload_to_memory and self.data_list is not None:
            data = self.data_list[index]
        else:
            data = torch.load(self.file_list[index])
        data = self.process(data)
        return data
    
    def process(self, data):
        sequence_data = []
        train_idx = [data['focal_idx']]
        
        # 'only_focal' for single-agent setting, 'focal_and_scored' for multi-agent setting
        if self.mode == 'focal_and_scored':
            train_idx += data['scored_idx']
        
        if self.split == "val":
            segments = [(self.val_sequence_start, 50)]
        elif self.split == "test":
            segments = [(0, 50)]
        else:
            segments = []
            for i in range(len(self.candidate_times)):
                for j in range(i+1, len(self.candidate_times)):
                    segments.append((self.candidate_times[i], self.candidate_times[j]))
        
        # 得到不同的时间步的训练数据
        for hist_start, hist_end in segments:
            for ag_idx in train_idx:
                ag_dict = self.process_single_agent(data, ag_idx, hist_start, hist_end)
                sequence_data.append(ag_dict)
        
        return sequence_data

    def process_single_agent(self, data, idx, hist_start, hist_end):
        hist_len = hist_end - hist_start  # 动态计算历史轨迹的长度  hist_end 作为当前时刻
        st, ed = hist_start, hist_end + self.num_future_steps
        cur_agent_id = data['agent_ids'][idx]
        origin = data['x_positions'][idx, hist_end - 1].double()
        theta = data['x_angles'][idx, hist_end - 1].double()
        rotate_mat = torch.tensor(
            [
                [torch.cos(theta), -torch.sin(theta)],
                [torch.sin(theta), torch.cos(theta)],
            ],
        )
        ag_mask = torch.norm(data['x_positions'][:, hist_end - 1] - origin, dim=-1) < self.radius
        ag_mask = ag_mask * data['x_valid_mask'][:, hist_end - 1]
        ag_mask[idx] = False
        valid_mask = data['x_valid_mask'][:, st: ed]
        valid_mask = torch.cat([valid_mask[[idx]], valid_mask[ag_mask]])

        # transform agents to local over the full window [0:ed] so we can expose earlier-history as target
        attr = torch.cat([data['x_attr'][[idx]], data['x_attr'][ag_mask]])
        pos_full = data['x_positions'][:, :ed]
        pos = torch.cat([pos_full[[idx]], pos_full[ag_mask]])
        head_full = data['x_angles'][:, :ed]
        head = torch.cat([head_full[[idx]], head_full[ag_mask]])
        vel_full = data['x_velocity'][:, :ed]
        vel = torch.cat([vel_full[[idx]], vel_full[ag_mask]])
        valid_mask_full = data['x_valid_mask'][:, :ed]
        valid_mask = torch.cat([valid_mask_full[[idx]], valid_mask_full[ag_mask]])

        # apply local transform using origin/theta at hist_end-1
        pos[valid_mask] = torch.matmul(pos[valid_mask].double() - origin, rotate_mat).to(torch.float32)
        head[valid_mask] = (head[valid_mask] - theta + np.pi) % (2 * np.pi) - np.pi

        # transform lanes to local
        l_pos = data['lane_positions']
        l_attr = data['lane_attr']
        l_is_int = data['is_intersections']
        l_pos = torch.matmul(l_pos.reshape(-1, 2).double() - origin, rotate_mat) \
                .reshape(-1, l_pos.size(1), 2).to(torch.float32)

        l_ctr = l_pos[:, 9:11].mean(dim=1)
        l_head = torch.atan2(
            l_pos[:, 10, 1] - l_pos[:, 9, 1],
            l_pos[:, 10, 0] - l_pos[:, 9, 0],
        )
        l_valid_mask = (
            (l_pos[:, :, 0] > -self.radius) & (l_pos[:, :, 0] < self.radius)
            & (l_pos[:, :, 1] > -self.radius) & (l_pos[:, :, 1] < self.radius)
        )

        l_mask = l_valid_mask.any(dim=-1)
        l_pos = l_pos[l_mask]
        l_is_int = l_is_int[l_mask]
        l_attr = l_attr[l_mask]
        l_ctr = l_ctr[l_mask]
        l_head = l_head[l_mask]
        l_valid_mask = l_valid_mask[l_mask]

        l_pos = torch.where(
            l_valid_mask[..., None], l_pos, torch.zeros_like(l_pos)
        )

        # remove outliers
        nearest_dist = torch.cdist(pos[:, hist_end - 1, :2], l_pos.view(-1, 2)).min(dim=1).values
        ag_mask2 = nearest_dist < 5

        ag_mask2[0] = True
        pos_all = pos[ag_mask2]
        head_all = head[ag_mask2]
        vel_all = vel[ag_mask2]
        attr = attr[ag_mask2]
        valid_all = valid_mask[ag_mask2]

        past_pos = pos_all[:, :hist_start] if hist_start > 0 else None
        past_vel = vel_all[:, :hist_start] if hist_start > 0 else None
        past_valid = valid_all[:, :hist_start] if hist_start > 0 else None

        pos_obs = pos_all[:, st:hist_end]
        head_obs = head_all[:, st:hist_end]
        vel_obs = vel_all[:, st:hist_end]
        valid_obs = valid_all[:, st:hist_end]

        head = head_obs
        vel = vel_obs
        pos_ctr = pos_obs[:, -1].clone()

        if hist_start > 0:
            type_mask = attr[:, [-1]] != 3
            target_mask = type_mask & valid_all[:, [hist_end - 1]] & past_valid
            target = torch.where(
                target_mask.unsqueeze(-1),
                past_pos - pos_ctr.unsqueeze(1),
                torch.zeros_like(past_pos),
            )
        else:
            target = target_mask = None

        diff_mask = valid_obs[:, :hist_len - 1] & valid_obs[:, 1:hist_len]
        tmp_pos = pos_obs.clone()
        pos = pos_obs
        pos_diff = pos[:, 1:hist_len] - pos[:, :hist_len - 1]

        target_diff = None
        if target is not None:
            target_diff_tmp = torch.cat((pos[:, -1].unsqueeze(1), target), dim=1)
            target_diff = target_diff_tmp[:, 1:hist_start + 1] - target_diff_tmp[:, :hist_start]
            target_diff_tmp = target_diff.clone()
            diff_mask_target_tmp = torch.cat((valid_obs[:, -1].unsqueeze(1), target_mask), dim=1)
            diff_mask_target = diff_mask_target_tmp[:, 1:hist_start + 1] & diff_mask_target_tmp[:, :hist_start]
            target_diff[:, :] = torch.where(
                diff_mask_target.unsqueeze(-1),
                target_diff_tmp,
                torch.zeros_like(target_diff_tmp),
            )

        target_vel_diff = None
        if past_vel is not None:
            target_vel_diff = torch.zeros_like(past_vel)
            if hist_start > 1:
                vel_past_delta = past_vel[:, 1:] - past_vel[:, :-1]
                vel_diff_mask = past_valid[:, 1:] & past_valid[:, :-1]
                target_vel_diff[:, 1:] = torch.where(
                    vel_diff_mask,
                    vel_past_delta,
                    torch.zeros_like(vel_past_delta),
                )

        pos[:, 1:hist_len] = torch.where(
            diff_mask.unsqueeze(-1),
            pos_diff,
            torch.zeros_like(pos_diff),
        )
        pos[:, 0] = torch.zeros(pos.size(0), 2, device=pos.device, dtype=pos.dtype)

        tmp_vel = vel.clone()
        vel_diff = vel[:, 1:hist_len] - vel[:, :hist_len - 1]
        vel[:, 1:hist_len] = torch.where(
            diff_mask,
            vel_diff,
            torch.zeros_like(vel_diff),
        )
        vel[:, 0] = torch.zeros(vel.size(0), device=vel.device, dtype=vel.dtype)

        return {
            'target': target,
            'target_diff': target_diff,
            'target_vel_diff': target_vel_diff,
            'target_mask': target_mask,

            'x_positions_diff': pos,
            'x_positions': tmp_pos,
            'x_attr': attr,
            'x_centers': pos_ctr,
            'x_angles': head,
            'x_velocity': tmp_vel,
            'x_velocity_diff': vel,
            'x_valid_mask': valid_obs,
            "hist_start": hist_start,
            "hist_end": hist_end,
            "hist_len": hist_len,

            'lane_positions': l_pos,
            'lane_centers': l_ctr,
            'lane_angles': l_head,
            'lane_attr': l_attr,
            'lane_valid_mask': l_valid_mask,
            'is_intersections': l_is_int,
            
            'origin': origin.view(1, 2),
            'theta': theta.view(1),
            'scenario_id': data['scenario_id'],
            'track_id': cur_agent_id,
            'city': data['city'],
            'timestamp': torch.Tensor([hist_end * 0.1])
        }
    

def collate_fn(seq_batch):
    seq_data = []
    for i in range(len(seq_batch[0])):
        batch = [b[i] for b in seq_batch]
        data = {}

        for key in [
            'x_positions_diff',
            'x_attr',
            'x_positions',
            'x_centers',
            'x_angles',
            'x_velocity',
            'x_velocity_diff',
            'lane_positions',
            'lane_centers',
            'lane_angles',
            'lane_attr',
            'is_intersections',
        ]:
            data[key] = pad_sequence([b[key] for b in batch], batch_first=True)

        if 'x_scored' in batch[0]:
            data['x_scored'] = pad_sequence(
                [b['x_scored'] for b in batch], batch_first=True
            )

        if batch[0]['target'] is not None:
            data['target'] = pad_sequence([b['target'] for b in batch], batch_first=True)
            data['target_diff'] = pad_sequence([b['target_diff'] for b in batch], batch_first=True)
            data['target_vel_diff'] = pad_sequence([b['target_vel_diff'] for b in batch], batch_first=True)
            data['target_mask'] = pad_sequence(
                [b['target_mask'] for b in batch], batch_first=True, padding_value=False
            )

        for key in ['x_valid_mask', 'lane_valid_mask']:
            data[key] = pad_sequence(
                [b[key] for b in batch], batch_first=True, padding_value=False
            )

        data['x_key_valid_mask'] = data['x_valid_mask'].any(-1)
        data['lane_key_valid_mask'] = data['lane_valid_mask'].any(-1)

        data['scenario_id'] = [b['scenario_id'] for b in batch]
        data['track_id'] = [b['track_id'] for b in batch]
        data['hist_start'] = torch.tensor([b['hist_start'] for b in batch])
        data['hist_end'] = torch.tensor([b['hist_end'] for b in batch])
        data['hist_len'] = torch.tensor([b['hist_len'] for b in batch])

        data['origin'] = torch.cat([b['origin'] for b in batch], dim=0)
        data['theta'] = torch.cat([b['theta'] for b in batch])
        data['timestamp'] = torch.cat([b['timestamp'] for b in batch])
        seq_data.append(data)
    return seq_data
