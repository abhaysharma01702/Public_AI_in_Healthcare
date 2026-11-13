"""Make 512 px copies of every prepared image.

Source: Data/Split/<dataset>/... (extracted, prepared copies of Data/Raw; several raw
datasets are still zipped). Neither Split nor Raw nor any master CSV is modified.
Output: Data/Resized 512/<same relative path>.png, plus labels/resized_512_manifest.csv.

Per image:
  1. open with Pillow, apply the EXIF orientation tag, convert to RGB
  2. scale so the shorter side is exactly 512 px, aspect ratio kept, Lanczos
  3. SLID only: detect the black top-left square and fill it by harmonic (Laplace) inpainting
     with zero-gradient image borders (skimage's inpaint_biharmonic left a black
     wedge because the square touches the image border)
  4. save as PNG

Extra source: the 130 Nuclear Cataract images without an NC grade (IOL, ungraded, blood)
are not in Split; they are read from Data/Raw (extracted JPGs) and written to
Data/Resized 512/nuclear_cataract_no_grade/<split>/<image_id>.png (patient-level split from the
master CSV), for the quality-module pool only. Kept out of nuclear_cataract/<split>/<grade>/ so a
grading loader cannot pick them up as a class.

Excluded (listed in the manifest, no output file):
  - SLID second copies of the 8 pixel-identical pairs, checked pixel-identical here
  - SLIT-Net-v2 images with eval_include == False
  SLIT-Net is not processed at all (all its images are inside SLIT-Net v2).

Usage:
  python -I make_resized_512.py            # write images + manifest
  python -I make_resized_512.py --verify   # recompute every image in memory, compare MD5 with the files
  --only SET[,SET]                          # only these source sets (Split folder names or
                                            # nuclear_cataract_no_grade); other manifest rows are kept
Both modes append timing + environment to logs/run_log.jsonl (run_log.py).
"""
import hashlib
import io
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image, ImageFile, ImageOps
from scipy import ndimage, sparse
from scipy.sparse.linalg import spsolve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I does not add the script folder
from run_log import RunLog, folder_size  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True  # one SDCTrans JPEG is truncated in the release

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT = os.path.join(ROOT, "Data", "Split")
OUT = os.path.join(ROOT, "Data", "Resized 512")
MANIFEST = os.path.join(ROOT, "labels", "resized_512_manifest.csv")
SHORT = 512
GROW = 5  # px added around the detected square on every side (covers the soft / shadow edge)

# (dataset name, Split folder, master CSV)
DATASETS = [
    ("Nuclear Cataract", "nuclear_cataract", "nuclear_cataract_master.csv"),
    ("SUSTech-SYSU", "sustech_sysu", "sustech_master.csv"),
    ("SLID-E", "SLID-E", "SLID-E_master.csv"),
    ("Infectious Keratitis", "Infectious_Keratitis", "Infectious_Keratitis_master.csv"),
    ("SLID", "SLID", "SLID_master.csv"),
    ("SLIT-Net-v2", "SLIT-Net-v2", "SLIT-Net-v2_master.csv"),
    ("SDCTrans", "SDCTrans", "SDCTrans_master.csv"),
]

# second copy of each pixel-identical, far-apart pair -> first copy
SLID_COPIES = {
    "SLID_0116": "SLID_0003", "SLID_1298": "SLID_1211", "SLID_1299": "SLID_1212",
    "SLID_1300": "SLID_1213", "SLID_1301": "SLID_1214", "SLID_1451": "SLID_1411",
    "SLID_1599": "SLID_1585", "SLID_2400": "SLID_2386",
}


def load_rgb(path):
    im = Image.open(path)
    orientation = im.getexif().get(0x0112, 1)
    im = ImageOps.exif_transpose(im).convert("RGB")
    return im, orientation


def find_square(a):
    """Black rectangle at the top-left corner of a SLID image (float RGB in [0, 1]).

    Returns (x1, y1) = exclusive right / bottom edge in pixels, or None.
    Presence: the 4 x 4 px block at (2..5, 2..5) is black (max channel <= 2/255); checked a few
    pixels in because SLID_2085-2087 have a 1 px grey line in column 0 next to the square.
    Extent: a column (row) belongs to the square while >= 90% of its pixels in the top 10%
    (left 10%) of the image are dark (max channel <= 12/255), so the square's rim (values 1-6 in
    e.g. SLID_0864) is included; scanning starts at index 3 to step over an edge line.
    Search limits: 40% of the width, 25% of the height.
    """
    h, w = a.shape[:2]
    mx = a.max(axis=2)
    if not (mx[2:6, 2:6] <= 2 / 255.0).all():
        return None
    dark = mx <= 12 / 255.0
    # width limit 40%: SLID_2085-2087 have a wider bar (32.9% x 18.2% of the image)
    col_frac = dark[: int(0.10 * h), 3: int(0.40 * w)].mean(axis=0)
    row_frac = dark[3: int(0.25 * h), : int(0.10 * w)].mean(axis=1)
    x1 = 3 + (int(np.argmax(col_frac < 0.9)) if (col_frac < 0.9).any() else col_frac.size)
    y1 = 3 + (int(np.argmax(row_frac < 0.9)) if (row_frac < 0.9).any() else row_frac.size)
    return x1, y1


def _find_square_v1(a):
    """Detector used (kept for the record; replaced )."""
    h, w = a.shape[:2]
    dark = a.max(axis=2) <= 2 / 255.0
    if not dark[:3, :3].all():
        return None
    col_frac = dark[: int(0.10 * h), : int(0.25 * w)].mean(axis=0)
    row_frac = dark[: int(0.25 * h), : int(0.10 * w)].mean(axis=1)
    x1 = int(np.argmax(col_frac < 0.9)) if (col_frac < 0.9).any() else col_frac.size
    y1 = int(np.argmax(row_frac < 0.9)) if (row_frac < 0.9).any() else row_frac.size
    return x1, y1


def harmonic_fill(a, mask):
    """Fill masked pixels by solving Laplace's equation exactly (sparse solve).

    Each masked pixel equals the mean of its 4-neighbours that lie inside the image, so the
    image border acts as a zero-gradient (mirror) boundary and colours come only from real
    pixels around the mask. a: float array (H, W, C); mask: bool (H, W).
    """
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    n = len(ys)
    idx = -np.ones((h, w), int)
    idx[ys, xs] = np.arange(n)
    rows, cols = [], []
    rhs = np.zeros((n, a.shape[2]))
    deg = np.zeros(n)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ny, nx = ys + dy, xs + dx
        k = np.nonzero((ny >= 0) & (ny < h) & (nx >= 0) & (nx < w))[0]
        deg[k] += 1
        inner = mask[ny[k], nx[k]]
        rows.append(k[inner])
        cols.append(idx[ny[k[inner]], nx[k[inner]]])
        kk = k[~inner]
        rhs[kk] += a[ny[kk], nx[kk]]
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    A = sparse.csc_matrix((np.r_[deg, -np.ones(len(rows))], (np.r_[np.arange(n), rows], np.r_[np.arange(n), cols])), shape=(n, n))
    out = a.copy()
    out[ys, xs] = spsolve(A, rhs).reshape(n, -1)
    return out


def process(job):
    """Return (row dict, png bytes or None)."""
    row = dict(job)
    im, orientation = load_rgb(row["source_path"])
    W, H = im.size
    scale = SHORT / min(W, H)
    size = (round(W * scale), round(H * scale))
    if min(size) != SHORT:  # guard against rounding
        size = (SHORT, size[1]) if W <= H else (size[0], SHORT)
    small = im.resize(size, Image.LANCZOS)
    row.update(read_width=W, read_height=H, exif_orientation=orientation,
               resized_width=size[0], resized_height=size[1], scale=round(scale, 6),
               square_inpainted=False, square_x1=np.nan, square_y1=np.nan, square_dark_frac=np.nan)
    if row["dataset"] == "SLID":
        a = np.asarray(small).astype(np.float64) / 255.0
        box = find_square(a)
        if box is not None:
            x1, y1 = box
            row.update(square_x1=x1, square_y1=y1,
                       square_dark_frac=round(float((a[:y1, :x1].max(axis=2) <= 2 / 255.0).mean()), 4))
            mask = np.zeros(a.shape[:2], bool)
            mask[:y1, :x1] = True
            # grow as a rectangle (3 x 3 element), so the corners are covered too
            mask = ndimage.binary_dilation(mask, structure=np.ones((3, 3), bool), iterations=GROW)
            filled = harmonic_fill(a, mask)
            small = Image.fromarray((np.clip(filled, 0, 1) * 255).round().astype(np.uint8))
            row["square_inpainted"] = True
    buf = io.BytesIO()
    small.save(buf, format="PNG")
    data = buf.getvalue()
    row["md5"] = hashlib.md5(data).hexdigest()
    return row, data


def build_jobs():
    jobs, skipped = [], []
    for name, folder, master in DATASETS:
        m = pd.read_csv(os.path.join(ROOT, "labels", master)).set_index("image_id")
        n = 0
        for dirpath, _, files in os.walk(os.path.join(SPLIT, folder)):
            for fn in sorted(files):
                src = os.path.join(dirpath, fn)
                rel = os.path.relpath(src, SPLIT)
                image_id = os.path.splitext(fn)[0]
                split = rel.split(os.sep)[1]
                assert image_id in m.index, (name, image_id)
                assert m.at[image_id, "split"] == split or (split == "test" and m.at[image_id, "split"] == "external_test"), (image_id, split)
                row = dict(dataset=name, source_set=folder, image_id=image_id, split=m.at[image_id, "split"],
                           source_path=os.path.relpath(src, ROOT).replace(os.sep, "/"),
                           resized_path=("Data/Resized 512/" + os.path.splitext(rel)[0] + ".png").replace(os.sep, "/"),
                           orig_width=int(m.at[image_id, "orig_width"]), orig_height=int(m.at[image_id, "orig_height"]),
                           status="resized", note="")
                if name == "SLID" and image_id in SLID_COPIES:
                    row.update(status="excluded_duplicate_copy", note="pixel-identical copy of " + SLID_COPIES[image_id], resized_path="")
                    skipped.append(row)
                elif name == "SLIT-Net-v2" and not bool(m.at[image_id, "eval_include"]):
                    row.update(status="excluded_not_evaluated", note="eval_include False ", resized_path="")
                    skipped.append(row)
                else:
                    if name == "SDCTrans" and m.at[image_id, "status"] == "truncated_jpeg":
                        row["note"] = "truncated JPEG in release, decoded in tolerant mode "
                    jobs.append(row)
                n += 1
        print(f"{name:22s} {n:5d} files in Split")
    # Nuclear Cataract images without a grade, from the extracted raw JPGs
    m = pd.read_csv(os.path.join(ROOT, "labels", "nuclear_cataract_master.csv"))
    ng = m[m.status != "graded"].sort_values("image_id")
    for r in ng.itertuples():
        assert os.path.exists(os.path.join(ROOT, r.raw_path)), r.raw_path
        jobs.append(dict(dataset="Nuclear Cataract", source_set="nuclear_cataract_no_grade", image_id=r.image_id, split=r.split,
                         source_path=r.raw_path,
                         resized_path=f"Data/Resized 512/nuclear_cataract_no_grade/{r.split}/{r.image_id}.png",
                         orig_width=int(r.orig_width), orig_height=int(r.orig_height), status="resized",
                         note=f"no NC grade ({r.status}); quality-module pool only "))
    print(f"{'Nuclear Cataract (no grade)':22s} {len(ng):5d} files in Raw")
    return jobs, skipped


def check_slid_copies():
    """The 8 excluded SLID copies must be pixel-identical to the kept first copy."""
    for dup, first in SLID_COPIES.items():
        a = np.asarray(load_rgb(os.path.join(SPLIT, "SLID", _split_of("SLID", dup), dup + ".png"))[0])
        b = np.asarray(load_rgb(os.path.join(SPLIT, "SLID", _split_of("SLID", first), first + ".png"))[0])
        assert a.shape == b.shape and (a == b).all(), (dup, first)
    print("SLID: 8 excluded copies verified pixel-identical to their first copy")


def _split_of(ds, image_id):
    m = pd.read_csv(os.path.join(ROOT, "labels", "SLID_master.csv")).set_index("image_id")
    return m.at[image_id, "split"]


def main():
    verify = "--verify" in sys.argv
    only = set(sys.argv[sys.argv.index("--only") + 1].split(",")) if "--only" in sys.argv else None
    with RunLog("make_resized_512.py", notes="verify pass: recompute in memory, compare MD5" if verify else "write 512 px copies") as log:
        _main(verify, log, only)


def _main(verify, log, only):
    jobs, skipped = build_jobs()
    if only:
        jobs = [j for j in jobs if j["source_set"] in only]
        skipped = [j for j in skipped if j["source_set"] in only]
        log.add(only=sorted(only))
    check_slid_copies()
    print(f"{len(jobs)} images to process, {len(skipped)} excluded")
    rows, mismatches = [], 0
    workers = max(1, os.cpu_count() - 1)
    log.add(images_in=len(jobs) + len(skipped), images_out=len(jobs), images_excluded=len(skipped), workers=workers)
    with Pool(workers) as pool:
        for i, (row, data) in enumerate(pool.imap(process, jobs, chunksize=8), 1):
            dst = os.path.join(ROOT, row["resized_path"])
            if verify:
                with open(dst, "rb") as f:
                    if hashlib.md5(f.read()).hexdigest() != row["md5"]:
                        mismatches += 1
            else:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst, "wb") as f:
                    f.write(data)
            rows.append(row)
            if i % 1000 == 0:
                print(f"  {i} / {len(jobs)}", flush=True)
    df = pd.DataFrame(rows + skipped)
    cols = ["dataset", "source_set", "image_id", "split", "status", "source_path", "resized_path", "orig_width", "orig_height",
            "read_width", "read_height", "exif_orientation", "resized_width", "resized_height", "scale",
            "square_inpainted", "square_x1", "square_y1", "square_dark_frac", "md5", "note"]
    df = df[cols].sort_values(["dataset", "image_id"]).reset_index(drop=True)
    if verify:
        old = pd.read_csv(MANIFEST, keep_default_na=False)
        if only:
            old = old[old.source_set.isin(only)]
        old = old.sort_values(["dataset", "image_id"])
        same = (old["md5"].astype(str).tolist() == df["md5"].fillna("").astype(str).tolist())
        print(f"VERIFY: {mismatches} file MD5 mismatches; manifest md5 column identical: {same}")
        log.add(md5_mismatches=mismatches, manifest_md5_identical=bool(same))
        return
    if only:  # keep the other rows of the existing manifest
        old = pd.read_csv(MANIFEST)
        if "source_set" not in old:  # manifests written
            old.insert(1, "source_set", old.source_path.str.split("/").str[2])
        old = old[~old.source_set.isin(only)]
        df = pd.concat([old[cols], df]).sort_values(["dataset", "image_id"]).reset_index(drop=True)
    df.to_csv(MANIFEST, index=False)
    n_files, n_bytes = folder_size(OUT, ".png")
    log.add(output_files=n_files, output_bytes=n_bytes,
            per_dataset=df[df.status == "resized"].groupby("dataset").size().to_dict(),
            manifest_md5=hashlib.md5(open(MANIFEST, "rb").read()).hexdigest())

    # checks
    ok = df[df.status == "resized"]
    assert (ok[["resized_width", "resized_height"]].min(axis=1) == SHORT).all(), "shorter side not 512"
    assert (ok.scale <= 1).all(), "an image was enlarged"
    print("all shorter sides = 512; no image enlarged")
    print("EXIF orientation != 1:", int((ok.exif_orientation != 1).sum()))
    print("read size differs from master orig size:",
          int(((ok.read_width != ok.orig_width) | (ok.read_height != ok.orig_height)).sum()))
    print(df.groupby(["dataset", "status"]).size().to_string())
    s = ok[ok.dataset == "SLID"]
    print(f"SLID squares inpainted: {int(s.square_inpainted.sum)} / {len(s)}")
    print("square x1/y1 as % of width/height (min, median, max):")
    print(((s.square_x1 / s.resized_width) * 100).describe()[["min", "50%", "max"]].round(2).to_dict(),
          ((s.square_y1 / s.resized_height) * 100).describe()[["min", "50%", "max"]].round(2).to_dict())
    print("square dark fraction min:", s.square_dark_frac.min())
    log.add(slid_squares_inpainted=int(s.square_inpainted.sum()))


if __name__ == "__main__":
    main()
