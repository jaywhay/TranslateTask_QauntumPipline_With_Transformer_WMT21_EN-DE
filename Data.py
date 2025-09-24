# Data.py (clean, cache-driven, dual-tokenizer)
import os
from typing import Any, Dict, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from utils import load_cached_data
from ruamel.yaml import YAML

# ── 0) Config: single source of truth (configs/default.yaml) ────────────────
_yaml = YAML()
_CFG_PATH = os.path.join(os.path.dirname(__file__), "configs", "default.yaml")
with open(_CFG_PATH, "r", encoding="utf-8") as _f:
    _CFG = _yaml.load(_f)

_CACHE_PATH = os.path.normpath(_CFG["data"]["cache_path"])

def _abspath(path: str) -> str:
    if os.path.isabs(path):
        return path
    base = os.path.dirname(__file__)
    return os.path.normpath(os.path.join(base, path))

# ── 1) Cached dataframe loader ───────────────────────────────────────────────
def load_dataset():
    """
    prepare_data.py가 생성한 캐시를 로드.
    필요한 컬럼:
    parsed_source, parsed_target, dependency_info_en, dependency_info_de
    """
    cache_path = _abspath(_CACHE_PATH)
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Cache not found at {cache_path}. 먼저 `python prepare_data.py`로 캐시를 생성하세요."
        )
    df = load_cached_data(cache_path)
    required = {"parsed_source", "parsed_target", "dependency_info_en", "dependency_info_de"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Cached dataframe is missing columns: {missing}. "
                    "prepare_data.py 최신 버전으로 다시 생성하세요.")
    return df

# ── 2) Dataset (dual-tokenizer; src/tgt may differ) ─────────────────────────
class TranslationDataset(Dataset):
    def __init__(self, df, tokenizer_src, tokenizer_tgt, max_length: int = 64):
        self.df = df.reset_index(drop=True)
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.df)

    def _ensure_tokens(self, item) -> List[str]:
        """spaCy 토큰/문자열 리스트 모두 안전 처리."""
        tokens = []
        try:
            for tok in item:
                tokens.append(tok.text if hasattr(tok, "text") else str(tok))
        except Exception:
            tokens = str(item).split()
        return tokens

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        src_tokens = self._ensure_tokens(row["parsed_source"])
        tgt_tokens = self._ensure_tokens(row["parsed_target"])

        enc = self.tokenizer_src(
            src_tokens, max_length=self.max_length, padding="max_length",
            truncation=True, is_split_into_words=True, return_tensors="pt",
        )
        dec = self.tokenizer_tgt(
            tgt_tokens, max_length=self.max_length, padding="max_length",
            truncation=True, is_split_into_words=True, return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": dec["input_ids"].squeeze(0),
            "dependency_info_en": row["dependency_info_en"],
            "dependency_info_de": row["dependency_info_de"],
        }

# ── 3) Collate (의존정보 리스트는 그대로 보존) ───────────────────────────────
def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in batch[0]:
        if key in ("dependency_info_en", "dependency_info_de"):
            out[key] = [b[key] for b in batch]
        else:
            out[key] = torch.stack([b[key] for b in batch], dim=0)
    return out

# ── 4) Loader 팩토리 ────────────────────────────────────────────────────────
def get_train_val_loaders(
    tokenizer_src,
    tokenizer_tgt,
    batch_size: Optional[int] = None,
    max_length: Optional[int] = None,
    val_size: float = 0.2,
    seed: Optional[int] = None,
):
    if batch_size is None:
        batch_size = int(_CFG.get("training", {}).get("batch_size", 32))
    if max_length is None:
        max_length = int(_CFG.get("training", {}).get("max_length", 64))
    if seed is None:
        seed = int(_CFG.get("training", {}).get("seed", 42))

    df = load_dataset()
    train_df, val_df = train_test_split(df, train_size=1 - val_size, random_state=seed)

    train_ds = TranslationDataset(train_df, tokenizer_src, tokenizer_tgt, max_length=max_length)
    val_ds   = TranslationDataset(val_df,   tokenizer_src, tokenizer_tgt, max_length=max_length)

    # --- DataLoader 튜닝 ---
    num_workers = max(1, os.cpu_count() // 2)
    pin_mem = torch.cuda.is_available()
    g = torch.Generator()
    g.manual_seed(seed)
    loader_kwargs = dict(
        batch_size=batch_size,
        collate_fn=_collate,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(train_ds, shuffle=True,  generator=g, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False,                 **loader_kwargs)
    return train_loader, val_loader
