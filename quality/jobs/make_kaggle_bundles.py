"""Build the two private Kaggle datasets for training the quality module.

1. Data/kaggle_upload/quality-data/quality_data.zip   `Data/Quality Data` as is (PNG; stored, not
   re-compressed: PNG is already compressed), top folder "Quality Data/".
2. Data/kaggle_upload/qm-code/qm_code.zip             quality/qm, train.py, evaluate.py, configs, jobs,
   data_prep/run_log.py, and the DenseNet121 ImageNet weights under torch_home/ (so the run needs no internet).
Each folder also gets dataset-metadata.json for `kaggle datasets create` (needs --user = Kaggle username).

Usage: python quality/jobs/make_kaggle_bundles.py --user <kaggle_username> [--only code|data]
"""
import argparse
import hashlib
import json
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "data_prep"))
from run_log import RunLog  # noqa: E402

V2 = "--v2" in sys.argv                     # Quality Data v2 -> dataset slitlamp-quality-data-v2
QD = os.path.join(ROOT, "Data", "Quality Data v2" if V2 else "Quality Data")
OUT = os.path.join(ROOT, "Data", "kaggle_upload")
WEIGHTS = "densenet121-a639ec97.pth"
CODE_FILES = ["quality/qm/__init__.py", "quality/qm/data.py", "quality/qm/model.py", "quality/qm/losses.py",
              "quality/qm/metrics.py", "quality/qm/engine.py", "quality/train.py", "quality/evaluate.py",
              "quality/configs/densenet121.yaml", "quality/jobs/kaggle_train.py", "data_prep/run_log.py"]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def metadata(folder, user, slug, title):
    with open(os.path.join(folder, "dataset-metadata.json"), "w") as f:
        json.dump({"title": title, "id": f"{user}/{slug}", "licenses": [{"name": "unknown"}]}, f, indent=1)


def build_data(user):
    folder = os.path.join(OUT, "quality-data-v2" if V2 else "quality-data")
    os.makedirs(folder, exist_ok=True)
    z = os.path.join(folder, "quality_data.zip")
    files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(QD) for f in fs)
    with zipfile.ZipFile(z + ".tmp", "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        for i, p in enumerate(files):
            zf.write(p, os.path.join("Quality Data", os.path.relpath(p, QD)).replace(os.sep, "/"))
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{len(files)} files", flush=True)
    os.replace(z + ".tmp", z)
    metadata(folder, user, "slitlamp-quality-data-v2" if V2 else "slitlamp-quality-data",
             "Slit-lamp Quality Data v2" if V2 else "Slit-lamp Quality Data")
    return {"data_files": len(files), "data_zip_bytes": os.path.getsize(z), "data_zip_md5": md5(z)}


def build_code(user):
    folder = os.path.join(OUT, "qm-code")
    os.makedirs(folder, exist_ok=True)
    z = os.path.join(folder, "qm_code.zip")
    w = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints", WEIGHTS)
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in CODE_FILES:
            zf.write(os.path.join(ROOT, rel), "project/" + rel)
        zf.write(w, f"project/torch_home/hub/checkpoints/{WEIGHTS}")
    metadata(folder, user, "slitlamp-qm-code", "Slit-lamp quality module code")
    return {"code_files": len(CODE_FILES) + 1, "code_zip_bytes": os.path.getsize(z), "code_zip_md5": md5(z),
            "weights_md5": md5(w)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--only", choices=["code", "data"])
    ap.add_argument("--v2", action="store_true")
    a = ap.parse_args()
    with RunLog("quality/jobs/make_kaggle_bundles.py", notes="Kaggle upload bundles ") as log:
        res = {}
        if a.only != "data":
            res.update(build_code(a.user))
        if a.only != "code":
            res.update(build_data(a.user))
        print(json.dumps(res, indent=1))
        log.add(**res)


if __name__ == "__main__":
    main()
