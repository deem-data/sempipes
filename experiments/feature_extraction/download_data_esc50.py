from __future__ import annotations

import random
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

RANDOM_SEED = 42

TOTAL_SAMPLES = 10000

ESC50_REPO_ZIP_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"

TARGET_DIR = Path(f"tests/data/sounds-dataset-{TOTAL_SAMPLES}")
AUDIO_DIR = TARGET_DIR / "audio"
OUTPUT_CSV = TARGET_DIR / "sounds.csv"

BALANCED_PER_CLASS = False


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:  # nosec B310 (download utility)
        shutil.copyfileobj(r, f)


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def _balanced_sample_indices_by_category(df: pd.DataFrame, n: int, seed: int) -> list[int]:
    """
    Sample (approximately) equally across categories.
    """
    rng = random.Random(seed)
    categories = sorted(df["category"].dropna().unique().tolist())
    if not categories:
        return []

    per_class = n // len(categories)
    remainder = n % len(categories)

    chosen: list[int] = []
    for cat in categories:
        idxs = df.index[df["category"] == cat].tolist()
        if not idxs:
            continue
        k = min(per_class + (1 if remainder > 0 else 0), len(idxs))
        if remainder > 0:
            remainder -= 1
        chosen.extend(rng.sample(idxs, k))

    # If we couldn't reach n because some classes had too few samples, top up uniformly.
    if len(chosen) < n:
        remaining = list(sorted(set(df.index.tolist()) - set(chosen)))
        if remaining:
            chosen.extend(rng.sample(remaining, min(n - len(chosen), len(remaining))))

    return chosen[:n]


with tempfile.TemporaryDirectory(prefix="esc50_download_") as tmp:
    tmp_path = Path(tmp)
    zip_path = tmp_path / "esc50.zip"
    _download(ESC50_REPO_ZIP_URL, zip_path)
    _extract_zip(zip_path, tmp_path)

    esc_root = tmp_path / "ESC-50-master"
    meta_csv = esc_root / "meta" / "esc50.csv"
    src_audio_dir = esc_root / "audio"

    if not meta_csv.exists():
        raise FileNotFoundError(f"Could not find ESC-50 metadata at {meta_csv}")
    if not src_audio_dir.exists():
        raise FileNotFoundError(f"Could not find ESC-50 audio directory at {src_audio_dir}")

    df = pd.read_csv(meta_csv)
    required_cols = {"filename", "fold", "target", "category", "esc10", "src_file", "take"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"ESC-50 metadata missing expected columns {missing}. Found columns: {list(df.columns)}")

    base_size = min(len(df), TOTAL_SAMPLES)
    print(
        f"Loaded ESC-50 with {len(df)} rows. "
        f"Base sample size={base_size}, target size={TOTAL_SAMPLES} (seed={RANDOM_SEED})."
    )

    if BALANCED_PER_CLASS:
        sampled_idx = _balanced_sample_indices_by_category(df, base_size, RANDOM_SEED)
        df_base = df.loc[sampled_idx].copy().reset_index(drop=True)
        print("Sampling strategy: approximately balanced per class.")
    else:
        df_base = df.sample(n=base_size, random_state=RANDOM_SEED).reset_index(drop=True)
        print("Sampling strategy: uniform over all rows.")

    # If the requested dataset is larger than ESC-50, repeat the base sample.
    if TOTAL_SAMPLES > len(df_base):
        repeat_times = TOTAL_SAMPLES // len(df_base)
        remainder = TOTAL_SAMPLES % len(df_base)
        parts = [df_base] * repeat_times
        if remainder:
            parts.append(df_base.sample(n=remainder, random_state=RANDOM_SEED).reset_index(drop=True))
        df_subsample = pd.concat(parts, ignore_index=True)
        print(
            f"Requested TOTAL_SAMPLES={TOTAL_SAMPLES} > ESC-50 size ({len(df)}). "
            f"Repeating base sample {repeat_times}x + {remainder} rows to reach {len(df_subsample)}."
        )
    else:
        df_subsample = df_base

    print("Category distribution:")
    print(df_subsample["category"].value_counts().sort_index())

    random.seed(RANDOM_SEED)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    rewritten_rows = []
    for i, row in enumerate(df_subsample.itertuples(index=False), start=1):
        src = src_audio_dir / str(row.filename)
        if not src.exists():
            raise FileNotFoundError(f"Missing wav file referenced by metadata: {src}")

        # Rename files to avoid collisions and keep a consistent naming scheme.
        dst_name = f"audio_{i:05d}{src.suffix.lower() or '.wav'}"
        dst = AUDIO_DIR / dst_name
        shutil.copy2(src, dst)

        new_row = {
            "filename": f"{TARGET_DIR}/audio/{dst_name}",
            "fold": int(row.fold),
            "target": int(row.target),
            "category": str(row.category),
            "esc10": bool(row.esc10),
            "src_file": str(row.src_file),
            "take": str(row.take),
        }
        rewritten_rows.append(new_row)

    out_df = pd.DataFrame(
        rewritten_rows, columns=["filename", "fold", "target", "category", "esc10", "src_file", "take"]
    )
    out_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved subsampled CSV to {OUTPUT_CSV}")
    print(f"Copied {len(out_df)} audio files to {AUDIO_DIR}")
