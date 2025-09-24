import optuna
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from Data import get_train_val_loaders
from main_model import QuantumFusionModel
import evaluate
import matplotlib.pyplot as plt

# Device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 고정 파라미터
INPUT_DIM = 64
OUTPUT_DIM = 30522 
BATCH_SIZE = 32 
EPOCHS_TUNE = 2
EPOCHS_FULL = 10
MAX_DECODE_LEN = 50

# 토크나이저 생성(양방향)
tokenizer_src = AutoTokenizer.from_pretrained("bert-base-german-cased")
tokenizer_tgt = AutoTokenizer.from_pretrained("bert-base-uncased")

# 평가 지표 로드
bleu_metric = evaluate.load("bleu")
try:
    comet_metric = evaluate.load("comet", model_id="unbabel/wmt21-comet-da")
except Exception:
    comet_metric = None

# 토큰 레벨 F1 계산 함수
def compute_token_f1(preds, refs):
    f1_scores = []
    for p, r in zip(preds, refs):
        pt, rt = p.split(), r.split()
        if not pt or not rt:
            f1_scores.append(0.0)
            continue
        c = set(pt) & set(rt)
        if not c:
            f1_scores.append(0.0)
        else:
            prec, rec = len(c)/len(pt), len(c)/len(rt)
            f1_scores.append(2*prec*rec/(prec+rec))
    return sum(f1_scores)/len(f1_scores) if f1_scores else 0.0

# 학습 및 검증 곡선 시각화 함수
def plot_history(hist):
    epochs = range(1, len(hist['train_loss'])+1)
    plt.figure()
    plt.plot(epochs, hist['train_loss'], label='Train Loss')
    plt.plot(epochs, hist['val_loss'],   label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.show()

    plt.figure()
    plt.plot(epochs, hist['val_bleu'], label='Val BLEU')
    if comet_metric:
        plt.plot(epochs, hist['val_comet'], label='Val COMET')
    plt.plot(epochs, hist['val_f1'], label='Val F1')
    plt.xlabel('Epoch'); plt.ylabel('Score'); plt.legend(); plt.show()

# Objective: 하이퍼파라미터 튜닝용, 손실만 반환
def objective(trial):
    # 하이퍼파라미터 샘플링
    n_qubits   = trial.suggest_int('n_qubits', 2, 6)
    n_layers   = trial.suggest_int('n_layers', 1, 4)
    hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128])
    lr         = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

    # 데이터 로더와 모델 초기화
    train_loader, val_loader = get_train_val_loaders(tokenizer_src, tokenizer_tgt, batch_size=BATCH_SIZE)
    model = QuantumFusionModel(INPUT_DIM, hidden_dim, OUTPUT_DIM, n_qubits, n_layers).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()), lr=lr
    )

    # 짧은 튜닝 루프: train/val 손실만 기록
    total_val_loss = 0.0

    for epoch in range(EPOCHS_TUNE):
        # TRAIN
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            deps = batch['dependency_info_en']
            loss = model(ids, labels, deps)
            loss.backward()
            optimizer.step()

        # VALIDATION
        model.eval()
        val_loss, count = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                deps = batch['dependency_info_en']
                loss = model(ids, labels, deps)
                val_loss += loss.item()
                count += ids.size(0)

        total_val_loss += (val_loss / count) if count else float('inf')

    return total_val_loss

if __name__ == '__main__':
    # 1) 하이퍼파라미터 튜닝
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    best   = study.best_trial.params
    print("=== Best Params ===", best)

    # 2) Full 학습 루프
    train_loader, val_loader = get_train_val_loaders(
        tokenizer_src, tokenizer_tgt, batch_size=BATCH_SIZE
        )
    model = QuantumFusionModel(
        input_dim=INPUT_DIM,
        hidden_dim=best['hidden_dim'],
        vocab_size=OUTPUT_DIM,
        n_qubits=best['n_qubits'],
        n_layers=best['n_layers']
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=best['lr'])
    criterion = nn.CrossEntropyLoss()

    # Full training history
    full_history = {k: [] for k in ['train_loss', 'val_loss', 'val_bleu', 'val_comet', 'val_f1']}

    for epoch in range(1, EPOCHS_FULL + 1):
        model.train()
        train_loss, total = 0.0, 0

        for batch in train_loader:
            optimizer.zero_grad()
            ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            deps = batch['dependency_info_en']
            loss = model(ids, labels, deps)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            total += ids.size(0)

        full_history['train_loss'].append(train_loss / total if total else float('inf'))


        # VALIDATION & METRICS
        model.eval()
        val_loss, total = 0.0, 0
        preds, refs, srcs = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                deps = batch['dependency_info_en']
                loss = model(ids, labels, deps)
                val_loss += loss.item()
                total += ids.size(0)

                # 디코딩
                decoded = model.decode(
                    ids, 
                    deps, 
                    tokenizer_tgt,
                    tokenizer_src, 
                    max_len=MAX_DECODE_LEN)
                for i, sample in enumerate(decoded):
                    srcs.append(sample['source'])
                    preds.append(sample['predicted'])
                    refs.append(tokenizer_tgt.decode(labels[i].tolist(), skip_special_tokens=True))

        full_history['val_loss'].append(val_loss / total if total else float('inf'))
        full_history['val_bleu'].append(bleu_metric.compute(predictions=preds, references=[[r] for r in refs])['bleu'])
        full_history['val_comet'].append(comet_metric.compute(sources=srcs, predictions=preds, references=refs).get('score', 0.0) if comet_metric else 0.0)
        full_history['val_f1'].append(compute_token_f1(preds, refs))

        print(f"Epoch {epoch}: Train={full_history['train_loss'][-1]:.4f}, Val={full_history['val_loss'][-1]:.4f}, "
            f"BLEU={full_history['val_bleu'][-1]:.4f}, COMET={full_history['val_comet'][-1]:.4f}, "
            f"F1={full_history['val_f1'][-1]:.4f}")

    # 결과 플롯
    plot_history(full_history)

    # 인퍼런스 데모 (autoregressive)
    print("\n=== Decoding Demo ===")
    batch = next(iter(val_loader))
    ids = batch['input_ids'].to(device)
    deps = batch['dependency_info_en']
    labels_inf = batch['labels'].to(device)

    decoded_samples = model.decode(ids, deps, tokenizer_tgt, tokenizer_src, max_len=MAX_DECODE_LEN)
    for i in range(len(decoded_samples)):
        decoded_samples[i]["reference"] = tokenizer_tgt.decode(labels_inf[i].tolist(), skip_special_tokens=True)

    # 문장별 BLEU 점수 계산
    sentence_bleus = [
        bleu_metric.compute(predictions=[s['predicted']], references=[[s['reference']]])['bleu']
        for s in decoded_samples
    ]
    avg_bleu = sum(sentence_bleus) / len(sentence_bleus)

    # 파일로 저장
    with open("decoding_samples.txt", "w", encoding="utf-8") as f:
        for i, sample in enumerate(decoded_samples):
            f.write(f"[Sample {i+1}]\n")
            f.write(f"Source   : {sample['source']}\n")
            f.write(f"Predicted: {sample['predicted']}\n")
            f.write(f"Reference: {sample['reference']}\n")
            f.write(f"BLEU     : {sentence_bleus[i] * 100:.2f}\n")
            f.write("-" * 60 + "\n")
        f.write(f"\nAverage Sentence BLEU: {avg_bleu * 100:.2f}\n")

