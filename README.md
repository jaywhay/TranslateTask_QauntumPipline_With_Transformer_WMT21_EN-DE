# TranslateTask_QauntumPipline_With_Transformer_WMT21_EN-DE
Hybrid Quantum–Classical NLP Architecture with QLSTM, QCNN, QGNN, and Transformer Heads — designed for optimization efficiency on large-scale machine translation tasks (WMT21 En-De).

Hybrid Quantum–Classical NLP
QLSTM, QCNN, QGNN, Transformer
Optimization / Efficiency focus
Dataset (WMT21 En-De)
GPU Simulation (A100, lightning.gpu, adjoint differentiation)

Python >= 3.10
PyTorch >= 2.0
PennyLane >= 0.34
transformers >= 4.41

# Clone
git clone https://github.com/username/repo.git
cd repo

# Install
pip install -r requirements.txt

# Run training
Be sure to run preparedata.py before running the code.
pyhthon preparedata.py
python train.py --config configs/train_config.yaml

@article{yourpaper2025,
  title={An Efficient Hybrid Quantum–classical NLP Architecture based on Structure Modular Design for Optimization},
  author={Jaeyun Jeong, James Park},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2025}
}
