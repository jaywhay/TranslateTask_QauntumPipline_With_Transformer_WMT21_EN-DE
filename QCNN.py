import torch
import torch.nn as nn
import pennylane as qml
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers
from utils import make_qdevice, make_qnode

class QCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_qubits, n_layers, config=None):
        super().__init__()
        self.n_qubits = n_qubits
        self.hidden_dim = hidden_dim
        self.config = config or {}
        self.mode = (self.config.get("model", {}) or {}).get("qcnn_pool", "attn")  # "attn" | "mean" | "chunk"
        self.chunk = int((self.config.get("model", {}) or {}).get("qcnn_chunk", 8))

        # 학습형 풀링: 토큰별 중요도 예측
        self.attn_score = nn.Linear(input_dim, 1)

        # QDevice/QNode
        self.dev = make_qdevice(self.config, wires=n_qubits)
        def circuit(inputs, weights):
            AngleEmbedding(inputs, wires=range(n_qubits))
            StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        qnode = make_qnode(self.dev, circuit, self.config)
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.quantum_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

        # 투영
        self.proj = nn.Linear(n_qubits, hidden_dim)
        self.global_proj = nn.Linear(hidden_dim, hidden_dim)

    def _attn_pool(self, x):
        # x: (S, D)
        score = self.attn_score(x).squeeze(-1)        # (S,)
        w = torch.softmax(score, dim=0).unsqueeze(-1) # (S,1)
        pooled = (w * x).sum(dim=0, keepdim=True)     # (1, D)
        return pooled

    def _chunk_pool(self, x):
        # x: (S, D) → 청크별 평균 후 (M, D), M=ceil(S/chunk)
        S = x.size(0)
        if S <= self.chunk:
            return x.mean(dim=0, keepdim=True)        # (1, D)
        pad = (self.chunk - (S % self.chunk)) % self.chunk
        if pad > 0:
            x = torch.cat([x, x.new_zeros(pad, x.size(1))], dim=0)  # zero-pad
        M = x.size(0) // self.chunk
        x = x.view(M, self.chunk, x.size(1)).mean(dim=1)            # (M, D)
        return x                                                    # (M, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (S, input_dim)
        if x.size(1) < self.n_qubits:
            raise ValueError(f"feature dim {x.size(1)} < n_qubits {self.n_qubits}")

        with torch.amp.autocast('cuda', enabled=False):
            if self.mode == "attn":
                pooled = self._attn_pool(x)                          # (1, D)
                q_in = pooled[:, :self.n_qubits].float()             # (1, Q)
                if (q_in.norm(p=2, dim=1) == 0).any():               # 안전 가드
                    q_in = q_in.clone(); q_in[0, 0] = 1.0
                q_out = self.quantum_layer(q_in)                     # (1, Q)

            elif self.mode == "chunk":
                chunks = self._chunk_pool(x)                         # (M, D)
                q_in = chunks[:, :self.n_qubits].float()             # (M, Q)
                zero_rows = (q_in.norm(p=2, dim=1) == 0)
                if zero_rows.any():
                    q_in = q_in.clone(); q_in[zero_rows, 0] = 1.0
                # TorchLayer는 배치 평가 지원 → 청크 M개를 한 번에
                q_out = self.quantum_layer(q_in)                     # (M, Q)
                q_out = q_out.mean(dim=0, keepdim=True)              # (1, Q)

            else:  # mean
                pooled = x.mean(dim=0, keepdim=True)                 # (1, D)
                q_in = pooled[:, :self.n_qubits].float()
                if (q_in.norm(p=2, dim=1) == 0).any():
                    q_in = q_in.clone(); q_in[0, 0] = 1.0
                q_out = self.quantum_layer(q_in)

        c_out = self.proj(q_out)                                     # (1, H)
        return self.global_proj(c_out)                               # (1, H)
          