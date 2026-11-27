import sys
from PIL import Image, ImageDraw, ImageFont
src, out = sys.argv[1], sys.argv[2]
box = tuple(int(v) for v in sys.argv[3].split(","))  # crop box in original pixels
im = Image.open(src).convert("RGB")
W, H = im.size
P = 420  # panel size for display
panels = [("Original (%d px short side)" % min(W, H), im.crop(box).resize((P, P), Image.NEAREST))]
for s in (1024, 512, 384, 224):
    f = s / min(W, H)
    small = im.resize((round(W * f), round(H * f)), Image.LANCZOS)
    b = tuple(round(v * f) for v in box)
    c = small.crop(b)
    panels.append(("Resized to %d  (crop = %d px wide)" % (s, c.size[0]), c.resize((P, P), Image.NEAREST)))
canvas = Image.new("RGB", (len(panels) * (P + 12) + 12, P + 60), "white")
d = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("arial.ttf", 17)
except OSError:
    font = ImageFont.load_default()
for i, (t, p) in enumerate(panels):
    x = 12 + i * (P + 12)
    canvas.paste(p, (x, 48))
    d.text((x, 16), t, fill="black", font=font)
canvas.save(out)
