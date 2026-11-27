"""Contact sheets for the visual review of screening candidates.

Usage: python -I screening_review_sheet.py <dataset> <flag> <out.png> [start=0] [n=40] [--crop]
--crop: show a 1:1 (unscaled) 220 x 165 px crop from the image centre instead of the whole image
        (thumbnails are too small to judge focus).
Shows flagged images of one dataset, worst first (by the flag's measure), from the 512 px copies,
labelled with a running number, image_id and the measure value.
"""
import os
import sys

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = {"flag_blur": ("sharp_z", True), "flag_dark": ("mean_lum", True),
         "flag_bright": ("mean_lum", False), "flag_over": ("frac_over", False)}

ds, flag, out = sys.argv[1], sys.argv[2], sys.argv[3]
start = int(sys.argv[4]) if len(sys.argv) > 4 else 0
n = int(sys.argv[5]) if len(sys.argv) > 5 else 40
crop = "--crop" in sys.argv
d = pd.read_csv(os.path.join(ROOT, "labels", "quality_screening_metrics.csv"))
m = pd.read_csv(os.path.join(ROOT, "labels", "resized_512_manifest.csv")).set_index("image_id")
col, asc = ORDER[flag]
sel = d[(d.dataset == ds) & d[flag]].sort_values(col, ascending=asc).iloc[start:start + n]
cols, tw, th = 8, 220, 165
rows = (len(sel) + cols - 1) // cols
sheet = Image.new("RGB", (cols * (tw + 6) + 6, rows * (th + 30) + 6), "white")
dr = ImageDraw.Draw(sheet)
try:
    font = ImageFont.truetype("arial.ttf", 12)
except OSError:
    font = ImageFont.load_default()
for i, r in enumerate(sel.itertuples()):
    im = Image.open(os.path.join(ROOT, m.at[r.image_id, "resized_path"])).convert("RGB")
    if crop:
        cx, cy = im.width // 2, im.height // 2
        im = im.crop((cx - tw // 2, cy - th // 2, cx - tw // 2 + tw, cy - th // 2 + th))
    else:
        im.thumbnail((tw, th))
    x, y = 6 + (i % cols) * (tw + 6), 6 + (i // cols) * (th + 30)
    dr.text((x, y), f"{start + i + 1}. {r.image_id[-22:]}", fill="black", font=font)
    dr.text((x, y + 13), f"{col}={getattr(r, col):.3g}", fill="black", font=font)
    sheet.paste(im, (x, y + 28))
sheet.save(out)
print(f"{ds} {flag}: {len(d[(d.dataset == ds) & d[flag]])} flagged; sheet shows {start + 1}-{start + len(sel)}")
