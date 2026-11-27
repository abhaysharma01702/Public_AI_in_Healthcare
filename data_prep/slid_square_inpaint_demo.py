"""Demo for what inpainting the SLID black square looks like.

Usage: python -I slid_square_inpaint_demo.py <raw SLID image> <output png>
Resizes to shorter side 512 (Lanczos, as proposed ), detects the black
top-left square (pixels <= 2 within the top-left 20% x 20%), and fills it with
scikit-image's biharmonic inpainting. Saves original | mask | inpainted side by side.
"""
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from skimage.restoration import inpaint_biharmonic

src, out = sys.argv[1], sys.argv[2]
im = Image.open(src).convert("RGB")
W, H = im.size
f = 512 / min(W, H)
im = im.resize((round(W * f), round(H * f)), Image.LANCZOS)
a = np.asarray(im).astype(np.float64) / 255.0
h, w = a.shape[:2]

# black pixels in the top-left corner region, keep the component touching (0, 0)
dark = (a.max(axis=2) <= 2 / 255.0)
corner = np.zeros_like(dark)
corner[: int(0.2 * h), : int(0.2 * w)] = True
lab, _ = ndimage.label(dark & corner)
mask = lab == lab[0, 0] if lab[0, 0] else np.zeros_like(dark)
# include the soft edge left by Lanczos resampling
mask = ndimage.binary_dilation(mask, iterations=3)
ys, xs = np.where(mask)
print("square (after dilation): x 0-%d, y 0-%d, %.2f%% of image" % (xs.max(), ys.max(), 100 * mask.mean()))

filled = inpaint_biharmonic(a, mask, channel_axis=-1)
filled = Image.fromarray((np.clip(filled, 0, 1) * 255).round().astype(np.uint8))
m = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")

panels = [("Resized (512), with black square", im), ("Detected square (white)", m), ("After inpainting", filled)]
gap, top = 12, 40
canvas = Image.new("RGB", (3 * w + 4 * gap, h + top + gap), "white")
d = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("arial.ttf", 18)
except OSError:
    font = ImageFont.load_default()
for i, (t, p) in enumerate(panels):
    x = gap + i * (w + gap)
    canvas.paste(p, (x, top))
    d.text((x, 10), t, fill="black", font=font)
canvas.save(out)
