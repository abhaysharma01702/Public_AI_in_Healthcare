"""Illustrate the screening sharpness measures.

Usage: python -I sharpness_demo.py <out.png> <image_id_sharp> <image_id_smooth>
Row 1: one image (512 px copy) and the same image blurred (Gaussian sigma 1, 2, 4 px); for each,
the 160 x 120 px centre crop and its Laplacian map (absolute values, brightened), with lap_var and
tenengrad. Row 2: an in-focus image with large smooth areas, which still scores low (content effect).
"""
import os
import sys
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out, sharp_id, smooth_id = sys.argv[1:4]
m = pd.read_csv(os.path.join(ROOT, "labels", "resized_512_manifest.csv")).set_index("image_id")


def measures(im):
    a = np.asarray(im).astype(np.float64)
    lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    lap = ndimage.laplace(lum)
    gx, gy = ndimage.sobel(lum, axis=1), ndimage.sobel(lum, axis=0)
    return lap, float(lap[2:-2, 2:-2].var()), float((gx ** 2 + gy ** 2)[2:-2, 2:-2].mean())


def tiles(im, title):
    lap, lv, tg = measures(im)
    w, h = im.size
    box = (w // 2 - 80, h // 2 - 60, w // 2 + 80, h // 2 + 60)
    crop = im.crop(box).resize((240, 180), Image.NEAREST)
    lmap = np.clip(np.abs(lap) * 6, 0, 255).astype(np.uint8)
    lmap = Image.fromarray(lmap).crop(box).resize((240, 180), Image.NEAREST).convert("RGB")
    return title, f"lap_var {lv:.1f} | tenengrad {tg:.0f}", crop, lmap


font = ImageFont.truetype("arial.ttf", 13) if os.name == "nt" else ImageFont.load_default()
base = Image.open(os.path.join(ROOT, m.at[sharp_id, "resized_path"])).convert("RGB")
row1 = [tiles(base, f"{sharp_id} (as is)")] + [tiles(base.filter(ImageFilter.GaussianBlur(s)), f"blurred, sigma {s} px") for s in (1, 2, 4)]
smooth = Image.open(os.path.join(ROOT, m.at[smooth_id, "resized_path"])).convert("RGB")
row2 = [tiles(smooth, f"{smooth_id} (in focus, smooth content)")]
W = 4 * 250 + 10
sheet = Image.new("RGB", (W, 2 * (2 * 180 + 50) + 20), "white")
d = ImageDraw.Draw(sheet)
for r, row in enumerate((row1, row2)):
    for c, (t, s, crop, lmap) in enumerate(row):
        x, y = 10 + c * 250, 10 + r * (2 * 180 + 50)
        d.text((x, y), t, fill="black", font=font)
        d.text((x, y + 15), s, fill="black", font=font)
        sheet.paste(crop, (x, y + 34))
        sheet.paste(lmap, (x, y + 34 + 182))
sheet.save(out)
