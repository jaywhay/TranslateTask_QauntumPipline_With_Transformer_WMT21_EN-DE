import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np
from utils import make_qdevice, make_qnode


class QuantumLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_qubits):
        super(QuantumLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        dev = qml.device("default.qubit", wires=n_qubits)
        weight_shapes = {"weights": (1, n_qubits, 3)}

        def quantum_lstm_gate(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        qnode = qml.QNode(quantum_lstm_gate, dev, interface="torch", diff_method="backprop")
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

        self.W_f = nn.Linear(n_qubits, hidden_dim)
        self.W_i = nn.Linear(n_qubits, hidden_dim)
        self.W_o = nn.Linear(n_qubits, hidden_dim)
        self.W_c = nn.Linear(n_qubits, hidden_dim)

    def forward(self, x, h_prev, c_prev):
        x_combined = torch.cat([x, h_prev], dim=1)  # (batch, input+hidden)

        if x_combined.dtype != torch.float32:
            x_combined = x_combined.float()

        x_transformed = x_combined[:, :self.n_qubits]
        x_cpu = x_transformed.detach().cpu()
        q_cpu = self.qlayer(x_cpu)
        q_output = q_cpu.to(x.device)

        f_t = torch.sigmoid(self.W_f(q_output))
        i_t = torch.sigmoid(self.W_i(q_output))
        o_t = torch.sigmoid(self.W_o(q_output))
        c_hat_t = torch.tanh(self.W_c(q_output))

        c_t = f_t * c_prev + i_t * c_hat_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


def create_graph_from_hidden_states(hidden_seq, dependency_pairs):
    """
    hidden_seq: torch.Tensor or np.ndarray, shape (S, hidden_dim)  # 단일 샘플의 토큰별 은닉 상태
    dependency_pairs: List[Tuple[int, int]]                        # (head_idx, dep_idx)
    Returns:
        nodes: List[np.ndarray]  # 각 토큰 노드 임베딩
        edges: List[List[int]]   # edge_index 형식으로 쓰기 쉽게 [src, dst] 쌍들의 리스트
    """
    # 텐서를 numpy로
    if hasattr(hidden_seq, "detach"):
        hs = hidden_seq.detach().cpu().numpy()
    elif hasattr(hidden_seq, "numpy"):
        hs = hidden_seq
    else:
        hs = np.asarray(hidden_seq)

    S = hs.shape[0]
    # 모든 토큰을 노드로 사용
    nodes = [hs[t] for t in range(S)]

    # 엣지 정리: 범위 밖/자가루프 제거
    edges = []
    for h, d in dependency_pairs:
        if 0 <= h < S and 0 <= d < S and h != d:
            edges.append([h, d])

    # 만약 비어있다면 시퀀스 인접선으로 최소 연결 보장
    if not edges and S > 1:
        edges = [[i - 1, i] for i in range(1, S)]

    return nodes, edges

class QLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_qubits, n_layers, config=None):
        super().__init__()
        self.n_qubits = n_qubits
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=1)
        self.to_qubits = nn.Linear(hidden_dim, n_qubits)
        self.to_hidden = nn.Linear(n_qubits, hidden_dim)
        self.config = config

        self.dev = make_qdevice(self.config, wires=n_qubits)

        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            for _ in range(n_layers):
                for w in range(n_qubits - 1):
                    qml.ISWAP(wires=[w, w + 1])
                for w in range(n_qubits):
                    qml.RZ(weights[w], wires=w)
            return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

        weight_shapes = {"weights": (n_qubits,)}
        qnode = make_qnode(self.dev, circuit, self.config)
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

    def forward(self, x):  # x: (B, S, input_dim)
        B, S, _ = x.shape
        h, _ = self.lstm(x)                      # (B, S, hidden_dim)
        z = self.to_qubits(h)                    # (B, S, n_qubits)

        # 시퀀스 각 토큰을 양자층에 통과(간단/안전: 토큰별 처리)
        outs = []
        with torch.amp.autocast('cuda', enabled=False):
          q_in  = z.reshape(B*S, self.n_qubits).float()
          q_out = self.qlayer(q_in)              # (B*S, n_qubits)
        qseq = q_out.view(B, S, self.n_qubits)            # (B, S, n_qubits)
        return self.to_hidden(qseq)              # (B, S, hidden_dim)
