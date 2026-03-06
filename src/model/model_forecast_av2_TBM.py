from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.lane_embedding import LaneEmbeddingLayer
from .layers.transformer_blocks import Block
from .layers.time_decoder import TimeDecoder
from .layers.mamba.vim_mamba import init_weights, create_block
from functools import partial
from timm.models.layers import DropPath, to_2tuple
from sklearn.preprocessing import StandardScaler
import os
import pickle
import numpy as np

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


# only 'DeMo'
class ModelForecast_qian(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        future_steps: int = 60,
    ) -> None:
        super().__init__()

        self.hist_embed_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, embed_dim),
        )

        # Agent Encoding Mamba
        self.hist_embed_mamba = nn.ModuleList(  
            [
                create_block(  
                    d_model=embed_dim,
                    layer_idx=i,
                    drop_path=0.2,  
                    bimamba=False,  
                    rms_norm=True,  
                )
                for i in range(4)
            ]
        )

        # backtrack-specific RMSNorms (layernorms differ between backtrack and predict models)
        self.back_norm_f10 = RMSNorm(embed_dim, eps=1e-5)
        self.back_norm_f20 = RMSNorm(embed_dim, eps=1e-5)
        self.back_norm_f30 = RMSNorm(embed_dim, eps=1e-5)
        self.back_norm_f40 = RMSNorm(embed_dim, eps=1e-5)

        self.drop_path = DropPath(drop_path)

        self.lane_embed = LaneEmbeddingLayer(3, embed_dim)

        self.pos_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Scene Context Transformer
        self.blocks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=0.2,
            )
            for i in range(5)
        )
        
        # backtrack-specific LayerNorms
        self.back_norm10 = nn.LayerNorm(embed_dim)
        self.back_norm20 = nn.LayerNorm(embed_dim)
        self.back_norm30 = nn.LayerNorm(embed_dim)
        self.back_norm40 = nn.LayerNorm(embed_dim)

        self.actor_type_embed = nn.Parameter(torch.Tensor(4, embed_dim))
        self.lane_type_embed = nn.Parameter(torch.Tensor(3, embed_dim))

        self.dense_predictor10 = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(), nn.Linear(256, 10 * 2)
        )
        self.dense_predictor20 = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(), nn.Linear(256, 20 * 2)
        )
        self.dense_predictor30 = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(), nn.Linear(256, 30 * 2)
        )
        self.dense_predictor40 = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(), nn.Linear(256, 40 * 2)
        )

        self.time_embedding_mlp = nn.Sequential(
            nn.Linear(1, 64), nn.GELU(), nn.Linear(64, embed_dim)
        )

        # create time decoders for different missing lengths (used for back-in-time reconstruction)
        # candidate missing lengths come from dataset candidate_times: 10,20,30,40,50
        self.time_decoder10 = TimeDecoder(future_len=10, dim=embed_dim)
        self.time_decoder20 = TimeDecoder(future_len=20, dim=embed_dim)
        self.time_decoder30 = TimeDecoder(future_len=30, dim=embed_dim)
        self.time_decoder40 = TimeDecoder(future_len=40, dim=embed_dim)

        self.features_dict = {}
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.actor_type_embed, std=0.02)
        nn.init.normal_(self.lane_type_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_from_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {
            k[len("net.") :]: v for k, v in ckpt.items() if k.startswith("net.")
        }
        return self.load_state_dict(state_dict=state_dict, strict=False)

    def _encode_hist_feat(self, hist_feat, hist_valid_mask, hist_len, norm_f_override=None):
        """
        Encode raw history features into agent embeddings using shared encoder (mamba blocks etc.)
        Returns: actor_feat_flat (B*N, C) and actor_feat (B, N, C)
        hist_feat: tensor [B, N, L, D]
        hist_valid_mask: tensor [B, N, L]
        hist_len: int
        """
        B, N, L, D = hist_feat.shape
        hist_feat_view = hist_feat.view(B * N, L, D)
        hist_key_valid_mask = hist_valid_mask.any(-1)
        hist_feat_key_valid = hist_key_valid_mask.view(B * N)

        # embed + mamba
        actor_feat = self.hist_embed_mlp(hist_feat_view[hist_feat_key_valid].contiguous())
        residual = None
        for blk_mamba in self.hist_embed_mamba:
            actor_feat, residual = blk_mamba(actor_feat, residual)

        # choose RMSNorm (or other) - allow override for backtrack vs predict
        norm_f = norm_f_override if norm_f_override is not None else getattr(self, f'norm_f{hist_len}')
        fused_add_norm_fn = rms_norm_fn if isinstance(norm_f, RMSNorm) else layer_norm_fn
        # when using RMSNorm-like object we need weight/bias attributes
        if isinstance(norm_f, RMSNorm):
            actor_feat = fused_add_norm_fn(
                self.drop_path(actor_feat),
                norm_f.weight,
                norm_f.bias,
                eps=norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=True,
            )
        else:
            # fallback: if norm_f is not RMSNorm-like, call fused function with placeholders
            actor_feat = fused_add_norm_fn(
                self.drop_path(actor_feat),
                getattr(norm_f, 'weight', None),
                getattr(norm_f, 'bias', None),
                eps=getattr(norm_f, 'eps', 1e-5),
                residual=residual,
                prenorm=False,
                residual_in_fp32=True,
            )
        actor_feat = actor_feat[:, -1]

        # restore to full B*N slots
        actor_feat_tmp = torch.zeros(
            B * N, actor_feat.shape[-1], device=actor_feat.device, dtype=actor_feat.dtype
        )
        actor_feat_tmp[hist_feat_key_valid] = actor_feat
        actor_feat_BN = actor_feat_tmp.view(B, N, actor_feat.shape[-1])

        return actor_feat_BN

    def forward(self, data):
        hist_start = data.get('hist_start')[0].item()
        hist_end = data.get('hist_end')[0].item()
        hist_len = data.get('hist_len')[0].item()
        
        ###### Scene context encoding (初始片段编码) ###### 
        # agent encoding: 原始输入片段的 raw 特征
        hist_valid_mask = data["x_valid_mask"]  # [B, N, L]
        hist_feat = torch.cat(
            [
                data["x_positions_diff"],
                data["x_velocity_diff"][..., None],
            ],
            dim=-1,
        )

        B, N, L, D = hist_feat.shape

        # 首次对当前输入片段进行编码，得到该片段的 agent embedding (使用 backtrack 的 norm_f)
        actor_feat = self._encode_hist_feat(
            hist_feat, hist_valid_mask, hist_len, norm_f_override=getattr(self, f'back_norm_f{hist_len}')
        )

        # 先做地图/位置的编码（提前），以便在每个回溯阶段都能构建完整的 scene-level 编码并保存
        lane_valid_mask = data["lane_valid_mask"]
        lane_normalized = data["lane_positions"] - data["lane_centers"].unsqueeze(-2)
        lane_normalized = torch.cat(
            [lane_normalized, lane_valid_mask[..., None]], dim=-1
        )
        B_l, M, L_l, D_l = lane_normalized.shape
        lane_feat = self.lane_embed(lane_normalized.view(-1, L_l, D_l).contiguous())
        lane_feat = lane_feat.view(B, M, -1)

        # type embedding and position embedding (提前计算)
        x_centers = torch.cat([data["x_centers"], data["lane_centers"]], dim=1)
        angles = torch.cat([data["x_angles"][:, :, -1], data["lane_angles"]], dim=1)
        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        pos_feat = torch.cat([x_centers, x_angles], dim=-1)
        pos_embed = self.pos_embed(pos_feat)

        actor_type_embed = self.actor_type_embed[data["x_attr"][..., 2].long()]
        lane_type_embed = self.lane_type_embed[data["lane_attr"][..., 0].long()]

        # 对初始片段构建 scene-level 编码并保存
        agent_encoder_init = actor_feat + actor_type_embed
        lane_feat_typed = lane_feat + lane_type_embed
        x_encoder_stage = torch.cat([agent_encoder_init, lane_feat_typed], dim=1)
        key_valid_mask_stage = torch.cat([
            hist_valid_mask.any(-1),
            data["lane_key_valid_mask"],
        ], dim=1)
        x_encoder_stage = x_encoder_stage + pos_embed
        norm_layer_bt = getattr(self, f'back_norm{hist_len}')
        for blk in self.blocks:
            x_encoder_stage = blk(
                x_encoder_stage,
                key_padding_mask=~key_valid_mask_stage,
                norm_layer=norm_layer_bt,
            )
        x_encoder_stage = norm_layer_bt(x_encoder_stage)
        self.features_dict[(hist_start, hist_end)] = x_encoder_stage[:, :N]

        hist_key_valid_mask = hist_valid_mask.any(-1)
        # If we have a missing prefix (hist_start > 0), predict the earlier/history frames
        # using a time decoder configured for the missing length. We no longer use
        # backtrack MLPs + re-encode full history; instead the decoder reconstructs
        # the earlier history in the same local coordinate frame.
        missing_len = hist_start
        if missing_len > 0:
            # pick decoder for this missing length; fallback to 50 if not found
            dec_attr = f'time_decoder{missing_len}'
            time_decoder = getattr(self, dec_attr, None)

        key_valid_mask = torch.cat([hist_key_valid_mask, data["lane_key_valid_mask"]], dim=1)

        ###### Trajectory decoding with decoupled queries ######
        new_y_hat = None
        new_pi = None
        dense_predict = None
        mode = None

        # outputs of other agents (kept as-is; these are not the main historical reconstructions)
        x_others = x_encoder_stage[:, 1:N]
        dense_predictor = getattr(self, f'dense_predictor{missing_len}', None)
        if dense_predictor is not None and missing_len > 0:
            y_hat_others = dense_predictor(x_others).view(B, x_others.size(1), -1, 2)
        else:
            y_hat_others = None

        # If we predicted a missing prefix, set decoder time length accordingly and run
        if missing_len > 0:
            device = x_encoder_stage.device
            time = torch.arange(missing_len, device=device).long()
            time = time * 0.1 + 0.1
            time = time.unsqueeze(-1)
            mode = self.time_embedding_mlp(time)
            mode = mode.repeat(x_encoder_stage.size(0), 1, 1)

            dense_predict, y_hat, pi, x_mode, new_y_hat, new_pi, mode_dense, scal, scal_new = \
                time_decoder(mode, x_encoder_stage, mask=~key_valid_mask)
            # y_hat/new_y_hat are reconstruction outputs for the earlier history (length=missing_len)
        else:
            # no missing prefix: return empty reconstructions
            dense_predict = None
            y_hat = None
            pi = None
            x_mode = None
            new_y_hat = None
            new_pi = None
            mode_dense = None
            scal = None
            scal_new = None

        ret_dict = {
            "y_hat": y_hat,  # trajectory output from mode query
            "pi": pi,  # probability output from mode query
            "scal": scal,  # output for Laplace loss from mode query

            "dense_predict": dense_predict,  # trajectory output from state query

            "y_hat_others": y_hat_others,  # trajectory of other agents

            "new_y_hat": new_y_hat,  # final trajectory output
            "new_pi": new_pi,  # final probability output     
            "scal_new": scal_new,  # final output for Laplace loss
        }

        return ret_dict