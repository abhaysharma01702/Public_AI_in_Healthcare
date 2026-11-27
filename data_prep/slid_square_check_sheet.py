"""Contact sheet for checking SLID black-square detection.

Usage: python -I slid_square_check_sheet.py <out.png> <image_id> [<image_id> ...]
For each image: top-left 40% x 40% of the 512 px copy BEFORE inpainting (resized from Data/Split)
with the detected box (red, from labels/resized_512_manifest.csv), and the same corner AFTER
inpainting (from Data/Resized 512).
"""
import sys
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

out, ids = sys.argv[1], sys.argv[2:]
m = pd.read_csv("labels/resized_512_manifest.csv").set_index("image_id")
tile = 260
sheet = Image.new("RGB", (2 * tile + 30, len(ids) * (tile + 26) + 10), "white")
d = ImageDraw.Draw(sheet)
try:
    font = ImageFont.truetype("arial.ttf", 15)
except OSError:
    font = ImageFont.load_default()
for i, iid in enumerate(ids):
    r = m.loc[iid]
    w, h = int(r.resized_width), int(r.resized_height)
    before = Image.open(r.source_path).convert("RGB").resize((w, h), Image.LANCZOS)
    after = Image.open(r.resized_path).convert("RGB")
    cw, ch = int(0.4 * w), int(0.4 * h)
    b = before.crop((0, 0, cw, ch)).copy()
    if r.square_inpainted in (True, "True"):
        ImageDraw.Draw(b).rectangle([0, 0, int(r.square_x1) - 1, int(r.square_y1) - 1], outline="red", width=2)
    y = 10 + i * (tile + 26)
    d.text((10, y), f"{iid}  box {r.square_x1:.0f} x {r.square_y1:.0f} of {w} x {h}", fill="black", font=font)
    f = tile / max(cw, ch)
    sheet.paste(b.resize((round(cw * f), round(ch * f))), (10, y + 20))
    sheet.paste(after.crop((0, 0, cw, ch)).resize((round(cw * f), round(ch * f))), (20 + tile, y + 20))
sheet.save(out)
