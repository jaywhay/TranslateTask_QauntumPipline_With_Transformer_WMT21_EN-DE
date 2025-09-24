import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from QLSTM import QLSTM, create_graph_from_hidden_states
from QGNN import QGNN
from QCNN import QCNN
from decoder import QuantumDecoder
import numpy as np
from typing import Optional, List
from attention import MultiHeadSelfAttention


# --- 상단 변경 없음 ---

class QuantumFusionModel(nn.Module):
    def __init__(self, config: dict, vocab_size_src: int, vocab_size_tgt: int, dropout: float = None):
        super().__init__()
        model_cfg = config.get('model', {})
        input_dim  = int(model_cfg.get('embed_dim', 64))
        hidden_dim = int(model_cfg.get('hidden_dim', 128))
        n_qubits   = int(model_cfg.get('qubits', 6))
        n_layers   = int(model_cfg.get('layers', 3))
        p_drop     = float(model_cfg.get('dropout', 0.3 if dropout is None else dropout))

        self.src_embedding = nn.Embedding(vocab_size_src, input_dim)

        self.qlstm = QLSTM(input_dim, hidden_dim, n_qubits, n_layers, config)
        self.qgnn  = QGNN(hidden_dim, hidden_dim, hidden_dim, n_qubits, n_layers, config)
        self.qcnn  = QCNN(hidden_dim, hidden_dim, n_qubits, n_layers, config)

        # Transformer head + 잔차용 LayerNorm
        tf_heads = int(model_cfg.get('tf_heads', 4))    # hidden_dim % tf_heads == 0
        self.tf_head  = MultiHeadSelfAttention(embed_dim=hidden_dim, num_heads=tf_heads, dropout=p_drop)
        self.enc_norm = nn.LayerNorm(hidden_dim)

        # fusion: [QGNN | QCNN | Attn] = 3H → H
        self.fusion = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=p_drop),
        )

        self.decoder = QuantumDecoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size_tgt,
            num_layers=n_layers,
        )
        self.n_layers = n_layers

    def forward(self, ids, dependency_info, labels=None):
      device = ids.device
      B, S = ids.size()

      # 1) Encoder
      embedded   = self.src_embedding(ids)          # (B, S, input_dim)
      hidden_seq = self.qlstm(embedded)             # (B, S, hidden_dim)

      # 2) Self-attn + Residual + LN
      attn_out = self.tf_head(hidden_seq)           # (B, S, hidden_dim)
      enc      = self.enc_norm(hidden_seq + attn_out)

      logits_list = []
      for i in range(B):
          x = enc[i]                                # (S_i, H)  ← 잔차 적용된 인코더 상태
          S_i = x.size(0)

          # 의존성 엣지 구성
          pairs = [(h, d) for (h, d) in dependency_info[i] if 0 <= h < S_i and 0 <= d < S_i and h != d]
          if not pairs and S_i > 1:
              pairs = [(j-1, j) for j in range(1, S_i)]
          eidx = torch.tensor(pairs, dtype=torch.long, device=device).t().contiguous()  # (2, E)

          gnn_out  = self.qgnn(Data(x=x, edge_index=eidx))          # (1, H)
          cnn_out  = self.qcnn(x)                                    # (1, H)
          attn_vec = attn_out[i].mean(dim=0, keepdim=True)           # (1, H) ← 헤드 출력 평균

          fused = torch.cat([gnn_out, cnn_out, attn_vec], dim=-1)    # (1, 3H)
          fusion_out = self.fusion(fused)                            # (1, H)

          if labels is None:
              raise ValueError("labels must be provided for training.")

          hidden_state = fusion_out.unsqueeze(0).repeat(self.n_layers, 1, 1)  # (L, 1, H)
          dec_in  = labels[i].unsqueeze(0)[:, :-1]                             # (1, S-1)
          logits_i, _ = self.decoder(dec_in, hidden_state)                     # (1, S-1, V)
          logits_list.append(logits_i)

      return torch.cat(logits_list, dim=0)                                      # (B, S-1, V)


    @torch.no_grad()
    def encode_only(self, ids, dependency_info):
      device = ids.device
      embedded   = self.src_embedding(ids)
      hidden_seq = self.qlstm(embedded)
      attn_out   = self.tf_head(hidden_seq)
      enc        = self.enc_norm(hidden_seq + attn_out)

      reps = []
      for i in range(ids.size(0)):
          x = enc[i]
          S_i = x.size(0)
          pairs = [(h, d) for (h, d) in dependency_info[i] if 0 <= h < S_i and 0 <= d < S_i and h != d]
          if not pairs and S_i > 1:
              pairs = [(j-1, j) for j in range(1, S_i)]
          eidx = torch.tensor(pairs, dtype=torch.long, device=device).t().contiguous()

          gnn_out  = self.qgnn(Data(x=x, edge_index=eidx))
          cnn_out  = self.qcnn(x)
          attn_vec = attn_out[i].mean(dim=0, keepdim=True)
          fused = torch.cat([gnn_out, cnn_out, attn_vec], dim=-1)
          reps.append(self.fusion(fused))                                   # (1, H)
      return reps

    @torch.no_grad()
    def decode(self, ids, dependency_info, max_len=64, bos_id=None, eos_id=None):
        device = ids.device
        embedded   = self.src_embedding(ids)
        hidden_seq = self.qlstm(embedded)
        attn_out   = self.tf_head(hidden_seq)
        enc        = self.enc_norm(hidden_seq + attn_out)

        outs = []
        for i in range(min(10, ids.size(0))):
            x = enc[i]
            S_i = x.size(0)
            pairs = [(h, d) for (h, d) in dependency_info[i] if 0 <= h < S_i and 0 <= d < S_i and h != d]
            if not pairs and S_i > 1:
                pairs = [(j-1, j) for j in range(1, S_i)]
            eidx = torch.tensor(pairs, dtype=torch.long, device=device).t().contiguous()

            gnn_out  = self.qgnn(Data(x=x, edge_index=eidx))
            cnn_out  = self.qcnn(x)
            attn_vec = attn_out[i].mean(dim=0, keepdim=True)

            fused = torch.cat([gnn_out, cnn_out, attn_vec], dim=-1)
            h = self.fusion(fused).unsqueeze(0).repeat(self.n_layers, 1, 1)     # (L,1,H)

            # 이후는 기존 디코더의 autoregressive/teacher-forcing 정책에 맞춰 사용
            # (현재 구현이 teacher forcing 기반이면 샘플 시퀀스 or BOS부터 rollout)
            # 필요 시 여기서 그리디/빔서치 구현 추가 가능
            outs.append(h)  # placeholder: 네 기존 decode 로직에 맞춰 연결
        return outs
