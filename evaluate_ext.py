# evaluate_ext.py
import torch, psutil, os, subprocess, time
from sacrebleu.metrics import CHRF, TER
import numpy as np

def compute_chrf(refs, hyps):
    metric = CHRF(word_order=2)
    return metric.corpus_score(hyps, [refs]).score

def compute_ter(refs, hyps):
    metric = TER()
    return metric.corpus_score(hyps, [refs]).score

def compute_energy_per_token(start_t, end_t, gpu_power_watts, n_tokens):
    elapsed = end_t - start_t
    energy_joules = gpu_power_watts * elapsed
    return energy_joules / max(1, n_tokens)

def sample_gpu_util(interval=0.5, samples=3):
    utils = []
    for _ in range(samples):
        try:
            out = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu",
                                        "--format=csv,noheader,nounits"])
            util = float(out.decode().strip().split("\n")[0])
            utils.append(util)
        except Exception:
            utils.append(0.0)
        time.sleep(interval)
    return np.mean(utils)

def gradient_variance(model):
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().flatten().cpu().numpy())
    if len(grads) == 0:
        return 0.0
    gcat = np.concatenate(grads)
    return float(np.var(gcat))

def extended_metrics(model, refs, hyps, gpu_power_watts=200.0, t_start=None, t_end=None, n_tokens=None):
    """
    통합 확장 평가: chrF++, TER, GPU util, Energy/token, Grad Var
    """
    metrics = {}
    metrics["chrF++"] = compute_chrf(refs, hyps)
    metrics["TER"] = compute_ter(refs, hyps)
    metrics["GPU_util(%)"] = sample_gpu_util()
    if t_start is not None and t_end is not None and n_tokens is not None:
        metrics["Energy/token(J)"] = compute_energy_per_token(t_start, t_end, gpu_power_watts, n_tokens)
    metrics["GradVar"] = gradient_variance(model)
    return metrics
