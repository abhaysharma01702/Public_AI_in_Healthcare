"""Generic contact sheet: python -I contact_sheet.py <out.png> <cols> <image paths...> (labels = file stems / parent folders)."""
import os
import sys
from PIL import Image, ImageDraw, ImageFont

out, cols, paths = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
tw, th = 300, 225
rows = (len(paths) + cols - 1) // cols
sheet = Image.new("RGB", (cols * (tw + 8) + 8, rows * (th + 26) + 8), "white")
d = ImageDraw.Draw(sheet)
try:
    font = ImageFont.truetype("arial.ttf", 13)
except OSError:
    font = ImageFont.load_default()
for i, p in enumerate(paths):
    im = Image.open(p).convert("RGB")
    im.thumbnail((tw, th))
    x, y = 8 + (i % cols) * (tw + 8), 8 + (i // cols) * (th + 26)
    d.text((x, y), "/".join(p.replace("\\", "/").split("/")[-3:])[:48], fill="black", font=font)
    sheet.paste(im, (x, y + 18))
sheet.save(out)
