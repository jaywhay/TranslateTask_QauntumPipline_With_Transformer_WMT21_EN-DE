import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import pennylane as qml
from utils import make_qdevice, make_qnode

class QGNN(MessagePassing):
    def __init__(self, in_dim, hidden_dim, out_dim, n_qubits, n_layers, config=None):
        super().__init__(aggr="add")
        self.n_qubits = n_qubits
        self.config = config or {}
        self.lin = nn.Linear(in_dim, n_qubits)
        self.post = nn.Linear(n_qubits, out_dim)
        self.attn_score = nn.Linear(n_qubits, 1)   # 학습형 그래프 풀링

        self.dev = make_qdevice(self.config, wires=n_qubits)
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            for l in range(weights.shape[0]):           # (L, Q)
                for w in range(n_qubits - 1):
                    qml.CNOT(wires=[w, w + 1])
                for w in range(n_qubits):
                    qml.RX(weights[l, w], wires=w)
            return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

        qnode = make_qnode(self.dev, circuit, self.config)
        weight_shapes = {"weights": (n_layers, n_qubits)}  # 레이어별 가중치
        self.quantum_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index               # x: (S, n_qubits)
        h = self.lin(x)                                       # (S, Q)
        aggr = self.propagate(edge_index, x=h)                # (S, Q)

        # 학습형 그래프 풀링
        score = self.attn_score(aggr).squeeze(-1)             # (S,)
        w = torch.softmax(score, dim=0).unsqueeze(-1)         # (S,1)
        pooled = (w * aggr).sum(dim=0, keepdim=True)          # (1, Q)

        with torch.amp.autocast('cuda', enabled=False):
            q_out = self.quantum_layer(pooled.float())        # (1, Q)
        return self.post(q_out)                               # (1, out_dim)

    def message(self, x_j):
        return x_j

