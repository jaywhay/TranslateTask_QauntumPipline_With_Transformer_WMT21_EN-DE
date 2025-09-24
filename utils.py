import os
import pandas as pd
import logging
from typing import Callable, Any
import pennylane as qml


def load_cached_data(cache_path: str) -> pd.DataFrame:
    """
    Load a cached DataFrame from pickle or parquet.
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache not found at {cache_path}")
    ext = os.path.splitext(cache_path)[1].lower()
    if ext in (".pkl", ".pickle"):
        return pd.read_pickle(cache_path)
    elif ext in (".parquet", ".parq"):
        return pd.read_parquet(cache_path)
    else:
        raise ValueError(f"Unsupported cache format: {ext}")


def save_cached_data(df: pd.DataFrame, cache_path: str) -> None:
    """
    Save a DataFrame to cache as pickle or parquet.
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    ext = os.path.splitext(cache_path)[1].lower()
    if ext in (".pkl", ".pickle"):
        df.to_pickle(cache_path)
    elif ext in (".parquet", ".parq"):
        df.to_parquet(cache_path)
    else:
        raise ValueError(f"Unsupported cache format: {ext}")


def parse_and_cache(
    raw_path: str,
    cache_path: str,
    parser: Callable[..., Any],
    text_column: str = "source",
    force_recompute: bool = False,
    **pipe_kwargs
) -> pd.DataFrame:
    
    if os.path.exists(cache_path) and not force_recompute:
        logging.info(f"Loading cached data from {cache_path}")
        return load_cached_data(cache_path)

    logging.info(f"Reading raw data from {raw_path}")
    df = pd.read_csv(raw_path)
    if text_column not in df.columns:
        raise KeyError(f"Column '{text_column}' not found in raw data")

    logging.info("Parsing text data via parser.pipe()...")
    texts = df[text_column].astype(str).tolist()
    parsed = list(parser.pipe(texts, **pipe_kwargs))
    df[f"parsed_{text_column}"] = parsed

    logging.info(f"Saving parsed data to cache at {cache_path}")
    save_cached_data(df, cache_path)

    return df

def make_qdevice(cfg, wires):
    be = cfg["quantum"]["backend"]       # default.qubit
    return qml.device(be, wires=wires)

def make_qnode(dev, circuit, cfg):
    return qml.QNode(
        circuit, 
        dev,
        interface="torch",
        diff_method=cfg["quantum"].get("diff_method", "adjoint")
    )
    