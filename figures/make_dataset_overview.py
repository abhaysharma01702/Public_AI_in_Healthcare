"""Figure: which lighting techniques each dataset contains (replaces the first version ).

All images are slit-lamp photographs; they differ in the LIGHTING TECHNIQUE used on the slit lamp:
    diffuse white light, slit beam (narrow white beam), cobalt blue + fluorescein, sclerotic scatter.
The figure is a lighting x dataset grid: an example photo where the dataset contains that lighting, an empty
cell where it does not, and the number of images per cell.

Where lighting comes from:
  - known from the release: SLIT-Net v2 and SDCTrans (folders / file names);
  - one lighting for the whole dataset, confirmed by a colour check on 150 random images plus a visual check
    of every image the colour rule flagged: Nuclear Cataract (slit beam), SUSTech-SYSU and
    SLID-E (cobalt blue), SLID (diffuse white);
  - Infectious Keratitis has no lighting labels: shares ESTIMATED from 150 random images with a colour
    rule, examples picked by the same rule and checked by eye.

Examples are picked with seed 14. Output: figures/dataset_overview.png
Usage:  python figures/make_dataset_overview.py
"""
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFile, ImageFont

ImageFile.LOAD_TRUNCATED_IMAGES = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures", "dataset_overview.png")
SEED = 14
FONT_DIR = "C:/Windows/Fonts"

TILE_W, TILE_H = 236, 177
GAP = 12
LABEL_W = 430
HEADER_H = 205
ROW_H = TILE_H + 40
COLS = ["Nuclear Cataract", "SUSTech-SYSU", "SLID-E", "Infectious Keratitis", "SLID", "SLIT-Net v2", "SDCTrans"]
EXTERNAL = {"SLIT-Net v2", "SDCTrans"}
LIGHTS = [
    ("Diffuse white light", "wide, even white light; the whole front of the eye is visible", "#F2F2F2"),
    ("Slit beam", "a narrow beam of white light cuts a thin optical section through cornea and lens", "#FFE08A"),
    ("Cobalt blue + fluorescein", "blue filter; the dye makes damaged surface or tear film glow green", "#3D6FD8"),
    ("Sclerotic scatter", "light aimed at the edge of the cornea, which then glows from within", "#E07A3D"),
]


def font(size, bold=False):
    return ImageFont.truetype(os.path.join(FONT_DIR, "arialbd.ttf" if bold else "arial.ttf"), size)


def csv(name):
    return pd.read_csv(os.path.join(ROOT, "labels", name))


def tile(rel_path):
    with Image.open(os.path.join(ROOT, rel_path)) as im:
        im.draft("RGB", (TILE_W * 3, TILE_H * 3))
        im = im.convert("RGB")
    w, h = im.size
    target = TILE_W / TILE_H
    if w / h > target:
        nw = int(h * target)
        im = im.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
    else:
        nh = int(w / target)
        im = im.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))
    return im.resize((TILE_W, TILE_H), Image.LANCZOS)


def colour_class(rel_path):
    """Simple colour rule: blue channel dominant = blue; dark = slit beam; else diffuse white."""
    with Image.open(os.path.join(ROOT, rel_path)) as im:
        im.draft("RGB", (160, 120))
        a = np.asarray(im.convert("RGB").resize((80, 60)), dtype=float)
    r, b, lum = a[..., 0].mean(), a[..., 2].mean(), a.mean()
    if b > r + 25:
        return "Cobalt blue + fluorescein"
    if lum < 55:
        return "Slit beam"
    return "Diffuse white light"


def cells():
    """{(lighting, dataset): (example path, count text)}"""
    out = {}
    nc = csv("nuclear_cataract_master.csv")
    nc = nc[nc["status"] == "graded"]
    out[("Slit beam", "Nuclear Cataract")] = (nc.sample(1, random_state=SEED + 3)["raw_path"].iloc[0], f"{len(nc):,}")

    su = csv("sustech_master.csv")
    su = su[su["status"] == "usable"]
    out[("Cobalt blue + fluorescein", "SUSTech-SYSU")] = (su[su["type"] == 3].sample(1, random_state=SEED)["raw_path"].iloc[0], f"{len(su):,}")

    se = csv("SLID-E_master.csv")
    out[("Cobalt blue + fluorescein", "SLID-E")] = (se[se["epiphora_stage"] == "mild"].sample(1, random_state=SEED)["split_path"].iloc[0], f"{len(se):,}")

    ik = csv("Infectious_Keratitis_master.csv")
    ik = ik[ik["status"] == "usable"]
    sample = ik.sample(150, random_state=SEED)
    classes = sample["split_path"].map(colour_class)
    for light in ("Diffuse white light", "Slit beam", "Cobalt blue + fluorescein"):
        paths = sample.loc[classes == light, "split_path"]
        share = 100 * (classes == light).mean()
        out[(light, "Infectious Keratitis")] = (paths.iloc[0], f"≈{share:.0f}% (est.)")

    sl = csv("SLID_master.csv")
    out[("Diffuse white light", "SLID")] = (sl[sl["lesions"].fillna("") == "Cataract"].sample(1, random_state=SEED)["split_path"].iloc[0], f"{len(sl):,}")

    v2 = csv("SLIT-Net-v2_master.csv")
    v2 = v2[v2["eval_include"]]
    for light, key in (("Diffuse white light", "diffuse_white"), ("Cobalt blue + fluorescein", "cobalt_blue")):
        sub = v2[(v2["illumination"] == key) & v2["in_SLIT-Net"]]
        out[(light, "SLIT-Net v2")] = (sub.sample(1, random_state=SEED)["split_path"].iloc[0], f"{(v2['illumination'] == key).sum:,}")

    sd = csv("SDCTrans_master.csv")
    sd = sd[sd["status"] == "usable"]
    for light, key in (("Diffuse white light", "diffuse_white"), ("Cobalt blue + fluorescein", "cobalt_blue"),
                       ("Sclerotic scatter", "sclerotic_scatter")):
        sub = sd[sd["illumination"] == key]
        out[(light, "SDCTrans")] = (sub.sample(1, random_state=SEED)["split_path"].iloc[0], f"{len(sub):,}")
    return out


def wrap(d, text, f, width):
    lines, cur = [], ""
    for word in text.split(" "):
        trial = (cur + " " + word).strip()
        if d.textlength(trial, font=f) <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    return lines + [cur]


def main():
    grid = cells()
    col_w = TILE_W + 2 * GAP
    W = LABEL_W + col_w * len(COLS) + GAP
    H = HEADER_H + ROW_H * len(LIGHTS) + 70
    fig = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(fig)

    d.text((GAP * 2, 16), "Lighting techniques in each dataset", font=font(38, True), fill="#111111")
    d.text((GAP * 2, 64), "All images are slit-lamp photographs; they differ in how the slit lamp lights the eye.",
           font=font(22), fill="#555555")

    n_train = len([c for c in COLS if c not in EXTERNAL])
    x0 = LABEL_W
    d.rectangle((x0, 112, x0 + col_w * n_train - 6, 119), fill="#2E7D32")
    d.rectangle((x0 + col_w * n_train, 112, x0 + col_w * len(COLS) - 6, 119), fill="#C62828")
    d.text((x0 + 4, 124), "training datasets (split 80 / 10 / 10)", font=font(19, True), fill="#2E7D32")
    d.text((x0 + col_w * n_train + 4, 124), "external test only", font=font(19, True), fill="#C62828")
    for c, name in enumerate(COLS):
        x = LABEL_W + c * col_w + GAP
        for k, line in enumerate(wrap(d, name, font(22, True), TILE_W)):
            d.text((x, 156 + 26 * k), line, font=font(22, True), fill="#111111")

    for r, (light, desc, colour) in enumerate(LIGHTS):
        y = HEADER_H + r * ROW_H
        d.rectangle((GAP, y, GAP + 10, y + TILE_H), fill=colour, outline="#BBBBBB")
        d.text((GAP + 24, y + 6), light, font=font(24, True), fill="#111111")
        for k, line in enumerate(wrap(d, desc, font(19), LABEL_W - 60)):
            d.text((GAP + 24, y + 42 + 24 * k), line, font=font(19), fill="#444444")
        for c, name in enumerate(COLS):
            x = LABEL_W + c * col_w + GAP
            if (light, name) in grid:
                path, count = grid[(light, name)]
                fig.paste(tile(path), (x, y))
                d.text((x + 2, y + TILE_H + 6), count + (" images" if "est." not in count else ""), font=font(18), fill="#333333")
            else:
                d.rectangle((x, y, x + TILE_W, y + TILE_H), fill="#F4F4F4", outline="#E2E2E2")
                d.text((x + TILE_W // 2, y + TILE_H // 2), "—", font=font(30), fill="#BBBBBB", anchor="mm")

    d.text((GAP * 2, H - 44), "Counts = images used. Infectious Keratitis has no lighting labels: shares estimated from 150 random images with a colour rule; examples checked by eye.",
           font=font(18), fill="#666666")
    fig.save(OUT, optimize=True)
    print(f"Wrote {OUT} ({W} x {H})")


if __name__ == "__main__":
    main()
