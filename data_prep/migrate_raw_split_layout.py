"""One-off migration to the Data/Raw + Data/Split layout (2026-10-12).

The folders were moved by hand (same drive, plain renames):
    Data/<dataset folder>        -> Data/Raw/<dataset folder>
    Data/slitnetv2.zip           -> Data/Raw/SLIT-Net-v2/slitnetv2.zip
    Data/SDCTrans Data.zip       -> Data/Raw/SDCTrans/SDCTrans Data.zip
    Data/splits                  -> Data/Split
    (Data/derived is unchanged)

This script rewrites the path prefixes in every CSV in labels/ by plain text replacement (only a value
that starts a CSV field is touched, so every other character of the file stays exactly as the build
scripts wrote it), then checks that every path in the `raw_path` / `split_path` columns exists. For
paths inside a zip ("<file>.zip/<member>") it checks that the zip exists and contains the member.

Idempotent: running it again on already-migrated CSVs changes nothing.

Usage: python data_prep/migrate_raw_split_layout.py
"""
import glob
import os
import re
import zipfile

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Old prefix -> new prefix, applied to the start of a CSV field. Each ends with "/", so "Data/SLID/"
# never matches "Data/SLID-E/...", and "Data/splits/" never matches "Data/Split/...".
PREFIXES = [
    ("Data/splits/", "Data/Split/"),
    ("Data/Nuclear Cataract Database/", "Data/Raw/Nuclear Cataract Database/"),
    ("Data/SUSTech-SYSU/", "Data/Raw/SUSTech-SYSU/"),
    ("Data/SLID-E/", "Data/Raw/SLID-E/"),
    ("Data/Infectious Keratitis Dataset (2025)/", "Data/Raw/Infectious Keratitis Dataset (2025)/"),
    ("Data/SLID/", "Data/Raw/SLID/"),
    ("Data/SLIT-Net/", "Data/Raw/SLIT-Net/"),
    ("Data/slitnetv2.zip/", "Data/Raw/SLIT-Net-v2/slitnetv2.zip/"),
    ("Data/SDCTrans/", "Data/Raw/SDCTrans/"),
]
PATH_COLUMNS = ("raw_path", "split_path")


def rewrite(text):
    count = 0
    for old, new in PREFIXES:
        # A field starts at the beginning of a line, after a comma, or after an opening quote.
        pattern = re.compile(r'(^|,|")' + re.escape(old), flags=re.MULTILINE)
        text, n = pattern.subn(lambda m: m.group(1) + new, text)
        count += n
    return text, count


def exists(path, zip_cache):
    if ".zip/" in path:
        zpath, member = path.split(".zip/", 1)
        zpath = os.path.join(ROOT, zpath + ".zip")
        if zpath not in zip_cache:
            zip_cache[zpath] = set(zipfile.ZipFile(zpath).namelist()) if os.path.exists(zpath) else set()
        return member in zip_cache[zpath]
    return os.path.exists(os.path.join(ROOT, path))


def main():
    zip_cache = {}
    for csv in sorted(glob.glob(os.path.join(ROOT, "labels", "*.csv"))):
        with open(csv, encoding="utf-8", newline="") as f:
            text = f.read()
        new_text, n = rewrite(text)
        if n:
            with open(csv, "w", encoding="utf-8", newline="") as f:
                f.write(new_text)
        df = pd.read_csv(csv, keep_default_na=False, dtype=str)
        cols = [c for c in PATH_COLUMNS if c in df.columns]
        missing = {c: [p for p in df[c] if p and not exists(p, zip_cache)] for c in cols}
        stale = sum(int(df[c].str.startswith(tuple(o for o, _ in PREFIXES)).sum()) for c in cols)
        print(f"{os.path.basename(csv)}: {n} paths rewritten; path columns {cols}; "
              f"missing files {sum(len(v) for v in missing.values)}; old-style paths left {stale}")
        for c, v in missing.items():
            for p in v[:5]:
                print(f"   missing {c}: {p}")


if __name__ == "__main__":
    main()
