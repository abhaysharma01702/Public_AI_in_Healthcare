"""Side-by-side sheet of image pairs (for visual duplicate checks).
Usage: python -I pair_sheet.py <out.png> <id_a>,<id_b> [<id_a>,<id_b> ...]   (ids from labels/resized_512_manifest.csv)
"""
import os
import sys
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out, pairs = sys.argv[1], [p.split(",") for p in sys.argv[2:]]
m = pd.read_csv(os.path.join(ROOT, "labels", "resized_512_manifest.csv")).set_index("image_id")
tw, th = 300, 225
sheet = Image.new("RGB", (2 * tw + 30, len(pairs) * (th + 24) + 10), "white")
d = ImageDraw.Draw(sheet)
try:
    font = ImageFont.truetype("arial.ttf", 13)
except OSError:
    font = ImageFont.load_default()
for i, (a, b) in enumerate(pairs):
    y = 8 + i * (th + 24)
    for j, iid in enumerate((a, b)):
        im = Image.open(os.path.join(ROOT, m.at[iid, "resized_path"])).convert("RGB")
        im.thumbnail((tw, th))
        x = 10 + j * (tw + 10)
        d.text((x, y), f"{iid} ({m.at[iid, 'split']})", fill="black", font=font)
        sheet.paste(im, (x, y + 16))
sheet.save(out)
