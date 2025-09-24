# prepare_data.py — Sharded + GPU + Resume + Range ops
import os, gc, re, math, argparse, sys
import pandas as pd
import spacy
import torch
from torch import amp
from tqdm import tqdm
from ruamel.yaml import YAML

# 파편화 완화
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def abspath(path: str) -> str:
    return os.path.normpath(path)

def parse_args():
    ap = argparse.ArgumentParser(description="Sharded, resume-safe spaCy parsing (EN/DE) with utilities")
    ap.add_argument("--lang", choices=["en", "de", "both"], default="both",
                    help="Which language to process")
    ap.add_argument("--start", type=int, default=0, help="Global start index (inclusive)")
    ap.add_argument("--end", type=int, default=None, help="Global end index (exclusive). Default: len(df)")
    ap.add_argument("--block-size", type=int, default=None, help="Override block size")
    ap.add_argument("--batch-size", type=int, default=None, help="Override parsing batch size")
    ap.add_argument("--resume", action="store_true", default=True, help="Skip existing parts (default True)")
    ap.add_argument("--force", action="store_true", help="Force reparse (delete existing parts in range)")
    ap.add_argument("--merge-only", action="store_true", help="Only merge existing parts, no parsing")
    ap.add_argument("--skip-merge", action="store_true", help="Skip final merge step")
    ap.add_argument("--clean-range", action="store_true", help="Delete en/de parts in the specified range and exit")
    ap.add_argument("--list-parts", action="store_true", help="List existing en/de parts and exit")
    return ap.parse_args()

def load_cfg():
    yaml = YAML()
    cfg_path = os.path.join(os.path.dirname(__file__), "configs", "default.yaml")
    cfg = yaml.load(open(cfg_path, "r", encoding="utf-8"))

    # preprocess 섹션이 없을 수도 있음
    pp = cfg.get("preprocess", {})
    en_model = pp.get("en_model", "en_core_web_trf")
    de_model = pp.get("de_model", "de_dep_news_trf")
    block_sz = int(pp.get("block_size", 50_000))
    batch_sz = int(pp.get("batch_size", 16))

    raw_path   = abspath(cfg["data"]["raw_path"])
    cache_path = abspath(cfg["data"]["cache_path"])
    cache_dir  = os.path.dirname(cache_path)
    os.makedirs(cache_dir, exist_ok=True)

    return cfg, dict(
        en_model=en_model, de_model=de_model,
        block_size=block_sz, batch_size=batch_sz,
        raw_path=raw_path, cache_path=cache_path, cache_dir=cache_dir
    )

def part_path(cache_dir: str, lang_prefix: str, s: int, e: int) -> str:
    return os.path.join(cache_dir, f"{lang_prefix}_part_{s:06d}_{e:06d}.pkl.gz")

def list_parts(cache_dir: str, lang_prefix: str):
    pat = re.compile(rf"^{re.escape(lang_prefix)}_part_(\d{{6}})_(\d{{6}})\.pkl.gz$")
    items = []
    for fn in os.listdir(cache_dir):
        m = pat.match(fn)
        if m:
            s = int(m.group(1)); e = int(m.group(2))
            items.append((s, e, os.path.join(cache_dir, fn)))
    return sorted(items, key=lambda x: x[0])

def clean_range(cache_dir: str, lang_prefixes, s: int, e: int, block_size: int):
    # 지정 범위를 샤딩 경계로 스냅
    s_snap = (s // block_size) * block_size
    e_snap = math.ceil((e if e is not None else s_snap) / block_size) * block_size
    removed = []
    for lp in lang_prefixes:
        parts = list_parts(cache_dir, lp)
        for ps, pe, fp in parts:
            if ps >= s_snap and pe <= e_snap and os.path.exists(fp):
                os.remove(fp)
                removed.append(fp)
    return removed

def parse_block(nlp, texts, bs: int):
    docs = []
    with torch.inference_mode(), amp.autocast('cuda', dtype=torch.float16):
        for doc in nlp.pipe(texts, batch_size=bs, n_process=1):
            docs.append(doc)
    return docs

def extract_deps(docs):
    return [[(tok.head.i, tok.i) for tok in doc if tok.dep_ != "punct"] for doc in docs]

def process_language(df, cache_dir, col_text, lang_prefix, model_name,
                     s_glob, e_glob, block, batch, resume=True, force=False):
    spacy.prefer_gpu()
    nlp = spacy.load(model_name, disable=["ner", "textcat"])

    N = len(df)
    if e_glob is None or e_glob > N:
        e_glob = N

    # 범위를 샤딩 경계로 정렬
    s_glob = (s_glob // block) * block
    e_glob = math.ceil(e_glob / block) * block
    e_glob = min(e_glob, N)

    for s in range(s_glob, e_glob, block):
        e = min(s + block, N)
        out_fp = part_path(cache_dir, lang_prefix, s, e)

        if force and os.path.exists(out_fp):
            os.remove(out_fp)

        if resume and os.path.exists(out_fp):
            # 이미 처리됨 → 스킵
            continue

        texts = df[col_text].iloc[s:e].tolist()
        docs = parse_block(nlp, texts, bs=batch)
        deps = extract_deps(docs)

        if lang_prefix == "en":
            part_df = pd.DataFrame({
                "source": df["source"].iloc[s:e].values,
                "target": df["target"].iloc[s:e].values,
                "dependency_info_en": deps
            })
        else:
            part_df = pd.DataFrame({
                "dependency_info_de": deps
            })

        part_df.to_pickle(out_fp, compression="gzip")

        # 메모리 정리
        del docs, deps, part_df, texts
        gc.collect(); torch.cuda.empty_cache()

    del nlp
    gc.collect(); torch.cuda.empty_cache()

def merge_parts(cache_dir: str, cache_path: str):
    print("[MERGE] collecting parts…")
    en_parts = list_parts(cache_dir, "en")
    de_parts = list_parts(cache_dir, "de")

    if not en_parts or not de_parts:
        raise RuntimeError("부분 캐시 부족: en_part_* / de_part_* 파일 확인 필요")

    if len(en_parts) != len(de_parts):
        raise RuntimeError(f"EN/DE 파트 개수 불일치: EN={len(en_parts)} DE={len(de_parts)}")

    merged_chunks = []
    for (s_en, e_en, en_fp), (s_de, e_de, de_fp) in zip(en_parts, de_parts):
        if (s_en, e_en) != (s_de, e_de):
            raise RuntimeError(f"범위 불일치: EN({s_en},{e_en}) != DE({s_de},{e_de})")
        en_df = pd.read_pickle(en_fp).reset_index(drop=True)  # .pkl.gz면 infer로 자동 인식
        de_df = pd.read_pickle(de_fp).reset_index(drop=True)
        merged_chunks.append(pd.concat([en_df, de_df], axis=1))
        del en_df, de_df
        gc.collect(); torch.cuda.empty_cache()

    # 먼저 정의
    final_cols = ["source", "target", "dependency_info_en", "dependency_info_de"]

    # (선택) 방어적 체크
    tmp = pd.concat(merged_chunks, axis=0).reset_index(drop=True)
    missing = [c for c in final_cols if c not in tmp.columns]
    if missing:
        raise RuntimeError(f"최종 캐시에 필요한 컬럼 누락: {missing}")

    # 이후 슬라이싱
    final_df = tmp[final_cols]

    final_df.to_pickle(cache_path, compression="gzip")
    print(f"Final cache saved: {cache_path}")

def main():
    args = parse_args()
    cfg, env = load_cfg()

    # 데이터 로드
    df = pd.read_csv(env["raw_path"]).dropna().drop_duplicates()
    df["source"] = df["source"].astype(str).str.strip().str.lower()
    df["target"] = df["target"].astype(str).str.strip().str.lower()

    # 파라미터 결정(override 지원)
    BLOCK = args.block_size if args.block_size else env["block_size"]
    BATCH = args.batch_size if args.batch_size else env["batch_size"]
    START = args.start
    END   = args.end

    # 관리 유틸: 목록 출력
    if args.list_parts:
        print("EN parts:", list_parts(env["cache_dir"], "en"))
        print("DE parts:", list_parts(env["cache_dir"], "de"))
        return

    # 관리 유틸: 구간 삭제
    if args.clean_range:
        removed = clean_range(env["cache_dir"], ["en", "de"], START, END if END else len(df), BLOCK)
        print("Removed parts:", removed)
        return

    # 병합만
    if args.merge_only:
        merge_parts(env["cache_dir"], env["cache_path"])
        return

    # 파싱
    if args.lang in ("en", "both"):
        print(f"[EN] model={env['en_model']}, block={BLOCK}, batch={BATCH}, range=({START},{END}) "
              f"{'(resume)' if args.resume and not args.force else ''}{'(force)' if args.force else ''}")
        process_language(df, env["cache_dir"], "source", "en", env["en_model"],
                         START, END, BLOCK, BATCH, resume=args.resume, force=args.force)

    if args.lang in ("de", "both"):
        print(f"[DE] model={env['de_model']}, block={BLOCK}, batch={BATCH}, range=({START},{END}) "
              f"{'(resume)' if args.resume and not args.force else ''}{'(force)' if args.force else ''}")
        process_language(df, env["cache_dir"], "target", "de", env["de_model"],
                         START, END, BLOCK, BATCH, resume=args.resume, force=args.force)

    # 병합 (원하면 생략)
    if not args.skip_merge:
        merge_parts(env["cache_dir"], env["cache_path"])

if __name__ == "__main__":
    main()



