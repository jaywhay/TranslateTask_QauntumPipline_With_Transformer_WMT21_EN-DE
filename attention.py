# attention.py

import torch
import torch.nn as nn
from typing import List

class AttentionHead(nn.Module):
    """
    Base interface for attention heads.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, embed_dim)
        Returns:
            Tensor of shape (batch_size, seq_len, head_dim)
        """
        raise NotImplementedError

class ClassicalHead(AttentionHead):
    """
    Standard scaled dot-product attention head.
    """
    def __init__(self, embed_dim: int, head_dim: int):
        super().__init__()
        self.query = nn.Linear(embed_dim, head_dim)
        self.key = nn.Linear(embed_dim, head_dim)
        self.value = nn.Linear(embed_dim, head_dim)
        self.scale = head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.query(x)                      # (batch, seq_len, head_dim)
        K = self.key(x)                        # (batch, seq_len, head_dim)
        V = self.value(x)                      # (batch, seq_len, head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, V)

class QuantumHead(AttentionHead):
    """
    Placeholder for quantum attention head.
    Implement quantum feature mapping and measurement here.
    """
    def __init__(self, embed_dim: int, head_dim: int, qubits: int, layers: int):
        super().__init__()
        # TODO: initialize quantum circuit parameters
        # e.g., self.q_nodes = build_quantum_layers(qubits, layers)
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: apply quantum circuit to input embeddings
        # return a tensor of shape (batch_size, seq_len, head_dim)
        raise NotImplementedError




class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads

        # ClassicalHead를 여러 개 병렬 사용
        from attention import ClassicalHead  # 동일 파일 내 클래스 재사용
        self.heads = nn.ModuleList([ClassicalHead(embed_dim, head_dim) for _ in range(num_heads)])
        self.out_proj = nn.Linear(num_heads * head_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, embed_dim)
        outs = [h(x) for h in self.heads]                     # 각: (B, S, head_dim)
        hcat = torch.cat(outs, dim=-1)                        # (B, S, num_heads*head_dim)
        y = self.out_proj(hcat)                               # (B, S, embed_dim)
        return self.dropout(y)