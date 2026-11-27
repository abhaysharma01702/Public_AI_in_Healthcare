"""Review sheet: for each image, the whole 512 px copy (thumbnail) and a 1:1 centre crop.

Usage: python -I image_pair_sheet.py <out.png> <image_id> [<image_id> ...]
Labels: running number, image_id, screening flags (from labels/quality_clean_pool.csv).
"""
import os
import sys
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out, ids = sys.argv[1], sys.argv[2:]
p = pd.read_csv(os.path.join(ROOT, "labels", "quality_clean_pool.csv")).set_index("image_id")
tw, th, per_row = 200, 150, 4
cell_w, cell_h = 2 * tw + 14, th + 32
rows = (len(ids) + per_row - 1) // per_row
sheet = Image.new("RGB", (per_row * cell_w + 10, rows * cell_h + 10), "white")
d = ImageDraw.Draw(sheet)
try:
    font = ImageFont.truetype("arial.ttf", 12)
except OSError:
    font = ImageFont.load_default()
for i, iid in enumerate(ids):
    r = p.loc[iid]
    im = Image.open(os.path.join(ROOT, r.resized_path)).convert("RGB")
    thumb = im.copy(); thumb.thumbnail((tw, th))
    cx, cy = im.width // 2, im.height // 2
    crop = im.crop((cx - tw // 2, cy - th // 2, cx + tw // 2, cy + th // 2))
    x, y = 10 + (i % per_row) * cell_w, 10 + (i // per_row) * cell_h
    reason = r.not_clean_reason if isinstance(r.not_clean_reason, str) else ""
    d.text((x, y), f"{i + 1}. {iid[-24:]}  [{reason}]", fill="black", font=font)
    sheet.paste(thumb, (x, y + 16))
    sheet.paste(crop, (x + tw + 4, y + 16))
sheet.save(out)
