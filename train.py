# train.py (refactored + AMP + GA + Clipping + BLEU)
import os, random, math, gzip, pickle, time
from typing import Dict, Any, List, Tuple
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer
from ruamel.yaml import YAML
from tqdm import tqdm
from evaluate_ext import extended_metrics
import matplotlib.pyplot as plt
import json


from Data import get_train_val_loaders
from main_model import QuantumFusionModel

# ------------ Paths ------------
gz_path   = "/content/drive/MyDrive/Colab Notebooks/NLP/cache/wmt21_parsed_lite.pkl.gz"
pkl_path  = gz_path.replace(".pkl.gz", ".pkl")
yaml_path = "/content/drive/MyDrive/Colab Notebooks/NLP/configs/default.yaml"

# .gz -> .pkl
if not os.path.exists(pkl_path):
    with gzip.open(gz_path, "rb") as f_in:
        df = pickle.load(f_in)
    with open(pkl_path, "wb") as f_out:
        pickle.dump(df, f_out)
    print(f"[OK] decompressed: {pkl_path}")
else:
    print(f"[SKIP] already exists: {pkl_path}")

yaml = YAML()
with open(yaml_path, "r", encoding="utf-8") as f:
    cfg = yaml.load(f)
cfg["data"]["cache_path"] = pkl_path
with open(yaml_path, "w", encoding="utf-8") as f:
    yaml.dump(cfg, f)
print(f"[OK] default.yaml updated: cache_path -> {pkl_path}")

# ------------ Utils ------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ids_to_text(tokenizer, ids_1d: List[int]) -> str:
    # 특수토큰 제거 + 공백 정규화
    return tokenizer.decode(ids_1d, skip_special_tokens=True).strip()

def batch_ids_to_text(tokenizer, ids_2d: torch.Tensor) -> List[str]:
    return [ids_to_text(tokenizer, row.tolist()) for row in ids_2d]

# --- Minimal BLEU-4 (single-reference), smoothing ---
def _ngram_counts(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)) if len(tokens) >= n else Counter()

def _prec_n(ref: List[str], hyp: List[str], n: int, eps=1e-9) -> float:
    ref_c = _ngram_counts(ref, n)
    hyp_c = _ngram_counts(hyp, n)
    overlap = sum(min(count, ref_c[ng]) for ng, count in hyp_c.items())
    total = max(1, sum(hyp_c.values()))
    return (overlap + eps) / (total + eps)

def compute_bleu(refs: List[str], hyps: List[str]) -> float:
    assert len(refs) == len(hyps) and len(refs) > 0
    # corpus-level BP + geometric mean of p1..p4
    ref_len = sum(len(r.split()) for r in refs)
    hyp_len = sum(len(h.split()) for h in hyps)
    BP = math.exp(1 - ref_len / hyp_len) if hyp_len < ref_len and hyp_len > 0 else 1.0

    # corpus-level n-gram precision (averaged via log)
    weights = [0.25, 0.25, 0.25, 0.25]
    sum_log_p = 0.0
    for n, w in zip([1,2,3,4], weights):
        # pool all n-grams across corpus
        overlap, total = 0.0, 0.0
        for r, h in zip(refs, hyps):
            p = _prec_n(r.split(), h.split(), n)
            overlap += math.log(max(p, 1e-12))
            total += 1.0
        avg_log_p = overlap / max(total, 1.0)
        sum_log_p += w * avg_log_p
    bleu = BP * math.exp(sum_log_p)
    return float(bleu) * 100.0  # sacreBLEU scale (0~100)

# --- Loss with label shift ---
def compute_loss(logits: torch.Tensor, labels: torch.Tensor, pad_id: int, *, where: str=""):
    """
    logits: (B, S-1, V), labels: (B, S)
    """
    assert logits.ndim == 3, f"[{where}] logits must be (B,S-1,V), got {logits.shape}"
    B, S_1, V = logits.shape
    assert labels.ndim == 2 and labels.size(1) == S_1 + 1, \
        f"[{where}] labels must be (B,S) with S = logits.S+1; got {labels.shape} vs {S_1+1}"
    target = labels[:, 1:]
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    return loss_fn(logits.reshape(-1, V), target.reshape(-1))

# --- Optional decode preview (try to be robust to return shape) ---
def decode_preview(model, batch, tokenizer_src, tokenizer_tgt, device, max_len=50, k=2) -> List[Tuple[str,str]]:
    ids_small  = batch["input_ids"][:k].to(device)
    deps_small = batch["dependency_info_en"][:k]
    # model.decode signature expected: (ids, deps, tokenizer_tgt, tokenizer_src, max_len=...)
    preds = model.decode(ids_small, deps_small, tokenizer_tgt, tokenizer_src, max_len=max_len)
    out = []
    # extract predictions
    if isinstance(preds, list):
        for i, p in enumerate(preds[:k]):
            if isinstance(p, dict):
                hyp = p.get("predicted", "")
                src = p.get("source", ids_to_text(tokenizer_src, ids_small[i].tolist()))
            elif isinstance(p, str):
                hyp = p
                src = ids_to_text(tokenizer_src, ids_small[i].tolist())
            else:
                hyp = str(p)
                src = ids_to_text(tokenizer_src, ids_small[i].tolist())
            out.append((src, hyp))
    return out

# ------------ train / eval core ------------
def run_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: optim.Optimizer,
    scheduler: CosineAnnealingLR,
    device: torch.device,
    pad_id_tgt: int,
    train: bool,
    epoch: int,
    *,
    use_amp: bool = True,
    grad_accum_steps: int = 1,
    clip_grad_norm: float = 0.0,
    tokenizer_src=None,
    tokenizer_tgt=None,
    decode_preview_every: int = 0,
    decode_max_len: int = 50,
    scaler: torch.cuda.amp.GradScaler = None,
):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_tokens = 0
    it = tqdm(dataloader, total=len(dataloader),
              desc=f"{'Train' if train else 'Val  '} {epoch}", ncols=120, dynamic_ncols=True)

    optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(train):
        for step, batch in enumerate(it, 1):
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            deps      = batch["dependency_info_en"]  # list of lists

            B = input_ids.size(0)
            # autocast: 양자 모듈 내부는 disabled, 비양자 경로만 amp 적용됨
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(input_ids, deps, labels)           # (B, S-1, V)
                loss   = compute_loss(logits, labels, pad_id_tgt, where="train" if train else "eval")

            # grad accumulation / clipping / step
            if train:
                if scaler is not None:
                    scaler.scale(loss / grad_accum_steps).backward()
                else:
                    (loss / grad_accum_steps).backward()

                do_step = (step % grad_accum_steps == 0) or (step == len(dataloader))
                if do_step:
                    if clip_grad_norm and clip_grad_norm > 0:
                        # unscale before clipping if using scaler
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * B
            total_tokens += B

            post = {"loss": f"{loss.item():.3f}"}
            if train:
                try:
                    post["lr"] = f"{scheduler.get_last_lr()[0]:.2e}"
                except Exception:
                    post["lr"] = f"{optimizer.param_groups[0]['lr']:.2e}"
            it.set_postfix(**post)

            # decode preview (train 중간 점검)
            if (train and decode_preview_every and (step % decode_preview_every == 0)
                and tokenizer_src is not None and tokenizer_tgt is not None):
                try:
                    model.eval()
                    with torch.no_grad():
                        previews = decode_preview(model, batch, tokenizer_src, tokenizer_tgt, device, max_len=decode_max_len, k=min(2, B))
                        for (src, hyp) in previews:
                            print(f"\n[SAMPLE] src: {src[:120]}")
                            print(f"[SAMPLE] hyp: {hyp[:120]}")
                finally:
                    model.train()

    avg = total_loss / max(1, total_tokens)
    return avg

def eval_bleu_subset(
    model: nn.Module,
    val_loader,
    tokenizer_src,
    tokenizer_tgt,
    device: torch.device,
    *,
    max_batches: int = 8,
    max_len: int = 50
) -> float:
    model.eval()
    refs, hyps = [], []
    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader, 1):
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            deps      = batch["dependency_info_en"]

            # 소량만 사용 (최대 16 샘플)
            take = min(16, input_ids.size(0))
            ids_small  = input_ids[:take]
            deps_small = deps[:take]

            preds = model.decode(ids_small, deps_small, tokenizer_tgt, tokenizer_src, max_len=max_len)
            # 예측 텍스트 추출
            pred_texts = []
            if isinstance(preds, list):
                for p in preds[:take]:
                    if isinstance(p, dict):
                        pred_texts.append(p.get("predicted", ""))
                    elif isinstance(p, str):
                        pred_texts.append(p)
                    else:
                        pred_texts.append(str(p))
            else:
                # 알 수 없는 형태면 스킵
                continue

            # 레퍼런스 텍스트는 labels에서 추출
            ref_texts = batch_ids_to_text(tokenizer_tgt, labels[:take])
            # 길이 맞추기
            m = min(len(pred_texts), len(ref_texts))
            refs.extend(ref_texts[:m])
            hyps.extend(pred_texts[:m])

            if b_idx >= max_batches:
                break

    if len(refs) == 0:
        return 0.0
    return compute_bleu(refs, hyps)

def plot_training_curves(log_history, save_dir="results"):
    """
    log_history: list of dicts [
        {"epoch": 1, "train_loss": 1.23, "val_loss": 1.10, 
        "gpu_util": 0.78, "energy_per_token": 0.92},
        ...
    ]
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = [d["epoch"] for d in log_history]
    train_loss = [d["train_loss"] for d in log_history]
    val_loss = [d["val_loss"] for d in log_history]
    gpu_util = [d["gpu_util"] * 100 for d in log_history]
    energy_tok = [d["energy_per_token"] for d in log_history]

    plt.figure(figsize=(6,4))
    plt.plot(epochs, train_loss, color="black", linewidth=2.2, label="Training Loss")
    plt.plot(epochs, val_loss, color="gray", linestyle="--", linewidth=2.2, label="Validation Loss")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.title("Training and Validation Loss", fontsize=12)
    plt.legend(fontsize=10, frameon=False)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig_loss_curve.png"), dpi=400, transparent=True)
    plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(epochs, gpu_util, color="black", linewidth=2.2, label="GPU Utilization (%)")
    plt.plot(epochs, energy_tok, color="gray", linestyle="--", linewidth=2.2, label="Energy per Token (J/token)")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Efficiency Metric", fontsize=11)
    plt.title("Efficiency Trend across Epochs", fontsize=12)
    plt.legend(fontsize=10, frameon=False)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig_efficiency_trend.png"), dpi=400, transparent=True)
    plt.close()
    print(f"[✓] Saved curves to {save_dir}/")


# ------------ main ------------
def main():
    # 1) Config / seed / device
    trn = cfg.get("training", {})
    opt = cfg.get("optimizer", {})
    sch = cfg.get("scheduler", {})

    set_seed(int(trn.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device)

    # 2) Tokenizers
    tok_name = trn.get("tokenizer", "Helsinki-NLP/opus-mt-en-de")
    tokenizer_src = AutoTokenizer.from_pretrained(tok_name)
    tokenizer_tgt = AutoTokenizer.from_pretrained(tok_name)
    if tokenizer_tgt.bos_token is None and tokenizer_tgt.cls_token is not None:
        tokenizer_tgt.bos_token = tokenizer_tgt.cls_token
    if tokenizer_tgt.eos_token is None and tokenizer_tgt.sep_token is not None:
        tokenizer_tgt.eos_token = tokenizer_tgt.sep_token
    if tokenizer_tgt.pad_token is None:
        tokenizer_tgt.pad_token = tokenizer_tgt.eos_token

    # 3) DataLoaders
    train_loader, val_loader = get_train_val_loaders(
        tokenizer_src, tokenizer_tgt,
        batch_size=int(trn.get("batch_size", 16)),
        max_length=int(trn.get("max_length", 64)),
        val_size=float(trn.get("val_size", 0.2)),
        seed=int(trn.get("seed", 42)),
    )

    # 4) Model / optim / sched
    vocab_src = len(tokenizer_src)
    vocab_tgt = len(tokenizer_tgt)
    model = QuantumFusionModel(cfg, vocab_size_src=vocab_src, vocab_size_tgt=vocab_tgt).to(device)
    pad_id_tgt = tokenizer_tgt.pad_token_id

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(opt.get("lr", 1e-4)),
        weight_decay=float(opt.get("weight_decay", 0.0))
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=int(sch.get("T_max", 10)))

    # AMP / GA / Clipping options
    use_amp          = bool(trn.get("mixed_precision", True)) and (device.type == "cuda")
    grad_accum_steps = int(trn.get("grad_accum_steps", 1))
    clip_grad_norm   = float(trn.get("clip_grad_norm", 0.0))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Decode preview & BLEU options
    decode_preview_every = int(trn.get("decode_preview_every", 0))     # e.g., 50
    decode_max_len       = int(trn.get("decode_max_len", 50))
    bleu_every           = int(trn.get("bleu_every", 1))               # 0=off, 1=every epoch
    bleu_samples_batches = int(trn.get("bleu_samples_batches", 4))     # val 배치 4개 * 16 = ~64샘플

    # 5) Loop
    epochs = int(trn.get("epochs", 10))
    best_val = float("inf")
    best_bleu = -1.0
    os.makedirs("checkpoints", exist_ok=True)
    log_history = []

    for ep in range(1, epochs + 1):
        t0 = time.time()
        train_loss = run_one_epoch(
            model, train_loader, optimizer, scheduler, device, pad_id_tgt,
            train=True, epoch=ep,
            use_amp=use_amp, grad_accum_steps=grad_accum_steps, clip_grad_norm=clip_grad_norm,
            tokenizer_src=tokenizer_src, tokenizer_tgt=tokenizer_tgt,
            decode_preview_every=decode_preview_every, decode_max_len=decode_max_len,
            scaler=scaler,
        )
        val_loss = run_one_epoch(
            model, val_loader, optimizer, scheduler, device, pad_id_tgt,
            train=False, epoch=ep,
            use_amp=False, grad_accum_steps=1, clip_grad_norm=0.0,
            tokenizer_src=None, tokenizer_tgt=None, scaler=None,
        )

        # 스케줄러 스텝(에폭 단위)
        scheduler.step()

        # 소규모 BLEU 평가
        bleu = None
        if bleu_every and (ep % bleu_every == 0):
            bleu = eval_bleu_subset(
                model, val_loader, tokenizer_src, tokenizer_tgt, device,
                max_batches=bleu_samples_batches, max_len=decode_max_len
            )
        t1 = time.time()
        ex_metrics = extended_metrics(model, refs=None, hyps=None,
                                    gpu_power_watts=200.0,
                                    t_start=t0, t_end=t1, n_tokens=bleu_samples_batches*16*64)
        print("[EXT_METRICS]", ex_metrics)
        
        log_entry = {
            "epoch": ep,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "gpu_util": ex_metrics.get("gpu_utilization", 0.0),
            "energy_per_token": ex_metrics.get("energy_per_token", 0.0),
        }
        log_history.append(log_entry)

        dt = time.time() - t0
        steps = len(train_loader)
        spb = dt / max(steps, 1)
        sps = (steps * train_loader.batch_size) / max(dt, 1e-9)
        if bleu is None:
            print(f"Epoch {ep}: train={train_loss:.4f} val={val_loss:.4f} | epoch_time={dt:.1f}s | sec/batch={spb:.2f} | samples/s={sps:.1f}")
        else:
            print(f"Epoch {ep}: train={train_loss:.4f} val={val_loss:.4f} BLEU={bleu:.2f} "
                f"| epoch_time={dt:.1f}s | sec/batch={spb:.2f} | samples/s={sps:.1f}")

        # 체크포인트 정책: val loss 기준 + BLEU 기준
        improved = False
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "checkpoints/best_val.pt")
            print("[CKPT] Saved best_val -> checkpoints/best_val.pt")
            improved = True
        if bleu is not None and bleu > best_bleu:
            best_bleu = bleu
            torch.save(model.state_dict(), "checkpoints/best_bleu.pt")
            print("[CKPT] Saved best_bleu -> checkpoints/best_bleu.pt")
            improved = True
        if not improved:
            # 필요하면 조기종료 로직을 여기 추가할 수 있음(patience 등)
            pass
    
    with open("results/training_log.json", "w") as f:
        json.dump(log_history, f, indent=2)
    plot_training_curves(log_history)
    print("Plots generated under results/ directory")
    
    torch.save(model.state_dict(), "checkpoints/final_model.pt")
    print("Training complete. Final model saved to checkpoints/final_model.pt")

if __name__ == "__main__":
    main()
