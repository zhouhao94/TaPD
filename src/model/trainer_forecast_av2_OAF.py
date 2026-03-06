import datetime
from pathlib import Path
import time
import pickle
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import defaultdict
from torchmetrics import MetricCollection
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.metrics import MR, minADE, minFDE, brier_minFDE
from src.utils.optim import WarmupCosLR
from src.utils.submission_av2 import SubmissionAv2
from src.utils.LaplaceNLLLoss import LaplaceNLLLoss
from omegaconf import OmegaConf

from .model_forecast_av2_TBM import ModelForecast_qian
from .model_forecast_av2_OAF import ModelForecast

class Trainer(pl.LightningModule):
    def __init__(
        self,
        model: dict,
        alignment_pairs=None,
        pretrained_weights: str = None,
        backtrack_weights: str = None,
        lr: float = 1e-3,
        warmup_epochs: int = 10,
        epochs: int = 60,
        weight_decay: float = 1e-4,
        align_weight_start: float = 0.05, 
        align_weight_end: float = 0.5,   
        align_weight_warmup_epochs: int = 10,
        freeze_backtrack: bool = True,
        isFinetune: bool = False,
    ) -> None:
        super(Trainer, self).__init__()

        if alignment_pairs is None:
            self.alignment_pairs = {}
        else:
            alignment_pairs = OmegaConf.to_container(alignment_pairs, resolve=True)
            new_alignment_pairs = {}
            for pair in alignment_pairs:
                new_alignment_pairs[tuple(pair[0])] = tuple(pair[1])
            self.alignment_pairs = new_alignment_pairs

        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()
        self.submission_handler = SubmissionAv2()

        self.align_weight_start = align_weight_start
        self.align_weight_end = align_weight_end
        self.align_weight_warmup_epochs = align_weight_warmup_epochs
        metrics = MetricCollection(
            {
                "minADE1": minADE(k=1),
                "minADE6": minADE(k=6),
                "minFDE1": minFDE(k=1),
                "minFDE6": minFDE(k=6),
                "MR": MR(),
                "b-minFDE6": brier_minFDE(k=6),
            }
        )
        self.laplace_loss = LaplaceNLLLoss()
        self.val_metrics = metrics.clone(prefix="val_")
        self.val_metrics_new = metrics.clone(prefix="val_new_")
        self.isFinetune = isFinetune
        self.freeze_backtrack = freeze_backtrack
        if self.isFinetune:
            self.net = ModelForecast_qian(**model)
            if backtrack_weights is not None:
                self.net.load_from_checkpoint(backtrack_weights)
                print('Backtrack weights have been loaded.')
            if self.freeze_backtrack:
                self.net.requires_grad_(False)
                self.net.eval()
        self.net_hou = ModelForecast(**model)

        if pretrained_weights is not None:
            self.net_hou.load_from_checkpoint(pretrained_weights)
            print('Forward model weights have been loaded.')

    def get_current_align_weight(self):
        current_epoch = self.current_epoch
        if current_epoch < self.align_weight_warmup_epochs:
            progress = current_epoch / self.align_weight_warmup_epochs
            weight = self.align_weight_start + \
                    0.5 * (self.align_weight_end - self.align_weight_start) * \
                    (1 - math.cos(progress * math.pi))
        else:
            weight = self.align_weight_end
        return weight
    
    def forward(self, data):
        if self.isFinetune:
            run_backtrack = True
            if self.freeze_backtrack:
                hist_start_tensor = data.get("hist_start")
                if hist_start_tensor is not None:
                    if hist_start_tensor.ndim == 0:
                        hist_start_tensor = hist_start_tensor.unsqueeze(0)
                    if torch.all(hist_start_tensor <= 0):
                        run_backtrack = False

            if run_backtrack:
                if self.freeze_backtrack:
                    self.net.eval()
                with torch.no_grad():
                    backtrack_out = self.net(data)
                    self._merge_backtrack_predictions(data, backtrack_out)
            else:
                backtrack_out = None
            out = self.net_hou(data)
        else:
            out = self.net_hou(data)
        return out

    def _merge_backtrack_predictions(self, data, backtrack_out):
        hist_start_tensor = data.get('hist_start')
        if hist_start_tensor is None:
            return
        if hist_start_tensor.ndim == 0:
            hist_start_tensor = hist_start_tensor.unsqueeze(0)
        hist_start = hist_start_tensor[0].item()
        if hist_start <= 0:
            return

        if torch.any(hist_start_tensor != hist_start_tensor[0]):
            raise RuntimeError("Mixed hist_start values within the same batch are not supported.")

        hist_len_tensor = data.get('hist_len')
        if hist_len_tensor is None:
            return
        hist_len = hist_len_tensor[0].item()
        total_len = hist_start + hist_len

        x_positions = data["x_positions"]
        x_velocity = data["x_velocity"]
        x_valid_mask = data["x_valid_mask"]

        B, N, _, _ = x_positions.shape
        new_y_hat = backtrack_out.get("new_y_hat")
        if new_y_hat is None:
            return

        pi = backtrack_out.get("new_pi")
        if pi is None:
            best_mode = torch.zeros(new_y_hat.size(0), dtype=torch.long, device=new_y_hat.device)
        else:
            best_mode = torch.argmax(pi, dim=-1)
        batch_idx = torch.arange(new_y_hat.size(0), device=new_y_hat.device)
        target_pred = new_y_hat[batch_idx, best_mode]
        target_pred = target_pred[..., :2]
        target_pred = target_pred.unsqueeze(1)

        pred_prefix = x_positions.new_zeros((B, N, hist_start, 2))
        pred_prefix[:, :1] = target_pred

        y_hat_others = backtrack_out.get("y_hat_others")
        if y_hat_others is not None and hist_start > 0:
            y_hat_others = y_hat_others[..., :2]
            num_fill = min(y_hat_others.size(1), max(0, N - 1))
            if num_fill > 0:
                pred_prefix[:, 1:1 + num_fill] = y_hat_others[:, :num_fill]

        full_positions = torch.cat([pred_prefix, x_positions], dim=2)
        data["x_positions"] = full_positions

        pos_diff_full = full_positions.new_zeros(full_positions.shape)
        pos_diff_full[:, :, 1:, :] = full_positions[:, :, 1:, :] - full_positions[:, :, :-1, :]
        data["x_positions_diff"] = pos_diff_full

        velocity_full = x_velocity.new_zeros((B, N, total_len))
        displacement = torch.norm(
            full_positions[:, :, 1:, :] - full_positions[:, :, :-1, :],
            dim=-1,
        ) / 0.1
        velocity_full[:, :, 1:] = displacement
        velocity_full[:, :, hist_start:] = x_velocity
        data["x_velocity"] = velocity_full

        velocity_diff_full = velocity_full.new_zeros(velocity_full.shape)
        velocity_diff_full[:, :, 1:] = velocity_full[:, :, 1:] - velocity_full[:, :, :-1]
        data["x_velocity_diff"] = velocity_diff_full

        agent_exists = x_valid_mask.any(dim=-1, keepdim=True)
        prefix_mask = agent_exists.expand(-1, -1, hist_start)
        full_valid_mask = torch.cat([prefix_mask, x_valid_mask], dim=-1)
        data["x_valid_mask"] = full_valid_mask
        data["x_key_valid_mask"] = full_valid_mask.any(-1)

    def predict(self, data):
        predictions = []
        probs = []
        for i in range(len(data)):
            cur_data = data[i]
            out = self(cur_data)
            prediction, prob = self.submission_handler.format_data(
                cur_data, out["y_hat"], out["pi"], inference=True)
            predictions.append(prediction)
            probs.append(prob)

        return predictions, probs

    def cal_loss(self, out, data, tag=''):
        y_hat, pi, y_hat_others = out["y_hat"], out["pi"], out["y_hat_others"]
        scal, scal_new = out["scal"], out["scal_new"]
        new_y_hat = out.get("new_y_hat", None)
        new_pi = out.get("new_pi", None)
        dense_predict = out.get("dense_predict", None)

        # gt
        y, y_others = data["target"][:, 0], data["target"][:, 1:]

        # loss for output of state query
        if dense_predict is not None:
            dense_reg_loss = F.smooth_l1_loss(dense_predict, y)
        else:
            dense_reg_loss = 0

        # loss for output of mode query
        l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
        best_mode = torch.argmin(l2_norm, dim=-1)
        y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]
        agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
        agent_cls_loss = F.cross_entropy(pi, best_mode.detach(), label_smoothing=0.2)
        
        # loss for final output
        if new_y_hat is not None:
            l2_norm_new = torch.norm(new_y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
            best_mode_new = torch.argmin(l2_norm_new, dim=-1)
            new_y_hat_best = new_y_hat[torch.arange(new_y_hat.shape[0]), best_mode_new]
            new_agent_reg_loss = F.smooth_l1_loss(new_y_hat_best[..., :2], y)
        else:
            new_agent_reg_loss = 0
        if new_pi is not None:
            new_pi_reg_loss = F.cross_entropy(new_pi, best_mode_new.detach(), label_smoothing=0.2)
        else:
            new_pi_reg_loss = 0

        # loss for other agents
        others_reg_mask = data["target_mask"][:, 1:]
        others_reg_loss = F.smooth_l1_loss(
            y_hat_others[others_reg_mask], y_others[others_reg_mask]
        )

        # Laplace loss, which is not necessary
        predictions = {}
        predictions['traj'] = y_hat
        predictions['scale'] = scal
        predictions['probs'] = pi
        laplace_loss = self.laplace_loss.compute(predictions, y)

        predictions['traj'] = new_y_hat
        predictions['scale'] = scal_new
        predictions['probs'] = new_pi
        laplace_loss_new = self.laplace_loss.compute(predictions, y)

        # total loss
        loss = agent_reg_loss + agent_cls_loss + others_reg_loss + \
                new_agent_reg_loss + dense_reg_loss + new_pi_reg_loss
        loss = loss + laplace_loss + laplace_loss_new

        disp_dict = {
            f"{tag}loss": loss.item(),
            f"{tag}reg_loss": agent_reg_loss.item(),
            f"{tag}cls_loss": agent_cls_loss.item(),
            f"{tag}others_reg_loss": others_reg_loss.item(),
            f"{tag}laplace_loss": laplace_loss.item(),
            f"{tag}laplace_loss_new": laplace_loss_new.item(),
        }
        if new_y_hat is not None:
            disp_dict[f"{tag}reg_loss_refine"] = new_agent_reg_loss.item()
        if new_pi is not None:
            disp_dict[f"{tag}reg_loss_new_pi"] = new_pi_reg_loss.item()
        if dense_predict is not None:
            disp_dict[f"{tag}reg_loss_dense"] = dense_reg_loss.item()

        return loss, disp_dict

    def training_step(self, data_list, batch_idx):
        total_loss = 0
        align_loss_total = 0
        total_loss_dict = defaultdict(float)
        
        # 根据输入轨迹长度分组,相同长度为一组 
        model_groups = {}
        for data in data_list:
            hist_len = data.get('hist_len')[0].item() 
            model_id = hist_len // 10 - 1  # av2: 10  av1: 5
            if model_id not in model_groups:
                model_groups[model_id] = []
            model_groups[model_id].append(data)
        
        self.net_hou.features_dict.clear()   # 对应模型保存的特征字典

        # 不同长度 由长到短
        for model_id in sorted(model_groups.keys(), reverse=True):
            model_data = sorted(
                model_groups[model_id],
                key=lambda d: d.get('hist_start')[0].item(),
            )
            num_samples = len(model_data)
            model_loss = 0

            for data in model_data:
                out = self(data)
                loss, loss_dict = self.cal_loss(out, data)
                model_loss += loss / num_samples
                for k, v in loss_dict.items():
                    total_loss_dict[k] += v / num_samples

            total_loss += model_loss / len(model_groups)  # 平均到不同长度模型

        # 特征损失
        align_count = 0
        align_loss = 0
        for (student_start, student_end), (teacher_start, teacher_end) in self.alignment_pairs.items():
            student_feat = self.net_hou.features_dict.get((student_start, student_end))
            teacher_feat = self.net_hou.features_dict.get((teacher_start, teacher_end))
            if student_feat is not None and teacher_feat is not None:
                align_weight = self.get_current_align_weight()
                with torch.no_grad():
                    teacher_feat_detached = teacher_feat.detach()
                align_loss = nn.L1Loss()(student_feat, teacher_feat_detached)
                align_loss_total += align_loss * align_weight
                align_count += 1

        if align_count > 0:
            align_loss_total /= align_count  

        total_loss += align_loss_total

        # 释放已不再需要的特征缓存，防止在多个 batch 间堆积
        self.net_hou.features_dict.clear()

        for k in total_loss_dict:
            self.log(
                f"train/{k}",
                total_loss_dict[k],
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        return total_loss

    def validation_step(self, data, batch_idx):
        if isinstance(data, list):
            data = data[0]
        out = self(data)
        _, loss_dict = self.cal_loss(out, data)
        metrics = self.val_metrics(out, data['target'][:, 0])
        if out['new_y_hat'] is not None:
            out['y_hat'] = out['new_y_hat']
            out['pi'] = out['new_pi']
        if out['new_y_hat'] is not None:
            metrics_new = self.val_metrics_new(out, data['target'][:, 0])

        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )
        if out['new_y_hat'] is not None:
            self.log_dict(
                metrics_new,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=1,
                sync_dist=True,
            )

    def on_test_start(self) -> None:
        save_dir = Path("./submission")
        save_dir.mkdir(exist_ok=True)
        self.submission_handler = SubmissionAv2(
            save_dir=save_dir
        )

    def test_step(self, data, batch_idx) -> None:
        if isinstance(data, list):
            data = data[0]
        out = self(data)
        if out['new_y_hat'] is not None:
            out['y_hat'] = out['new_y_hat']
            out['pi'] = out['new_pi']
        self.submission_handler.format_data(data, out["y_hat"], out["pi"])

    def on_test_end(self) -> None:
        self.submission_handler.generate_submission_file()

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-5,
            warmup_epochs=self.warmup_epochs,
            epochs=self.epochs,)

        return [optimizer], [scheduler]