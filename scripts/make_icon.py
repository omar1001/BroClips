"""Draws BroClips' icon once: broclips/web/broclips.ico (16-256 px) + broclips/web/broclips.png (256 px).
A bright gradient tile (the app's brand gradient: pink -> violet -> cyan) with a white play triangle that is
"jump-cut" in two. Run: python scripts/make_icon.py"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

S = 1024
OUT = Path(__file__).resolve().parents[1] / "broclips" / "web"


def draw():
    # diagonal gradient like the app's CSS (135deg): pink -> violet (55 %) -> cyan
    y, x = np.mgrid[0:S, 0:S].astype(np.float32)
    t = ((x + y) / (2 * (S - 1)))[..., None]
    stops = np.array([[255, 77, 141], [139, 92, 246], [34, 211, 238]], np.float32)
    rgb = np.where(t < 0.55, stops[0] + (stops[1] - stops[0]) * (t / 0.55),
                   stops[1] + (stops[2] - stops[1]) * ((t - 0.55) / 0.45))
    glow = np.exp(-(((x - S * 0.28) ** 2 + (y - S * 0.22) ** 2) / (2 * (S * 0.32) ** 2)))[..., None]
    rgb = rgb + (255 - rgb) * glow * 0.12                      # a soft light in the top-left corner
    tile = Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).convert("RGBA")
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle((24, 24, S - 25, S - 25), radius=230, fill=255)
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    img.paste(tile, (0, 0), mask)

    tri = [(372, 262), (372, 762), (802, 512)]
    shadow = Image.new("L", (S, S), 0)
    ImageDraw.Draw(shadow).polygon([(px + 14, py + 22) for px, py in tri], fill=110)
    shadow = shadow.filter(ImageFilter.GaussianBlur(26))
    img.paste(Image.new("RGBA", (S, S), (36, 10, 70, 255)), (0, 0), Image.fromarray(
        (np.asarray(shadow, np.float32) * (np.asarray(mask, np.float32) / 255)).astype(np.uint8)))

    white = Image.new("L", (S, S), 0)
    ImageDraw.Draw(white).polygon(tri, fill=255)
    cut = Image.new("L", (S, S), 0)                     # the "jump cut": a slanted gap through the triangle
    ImageDraw.Draw(cut).polygon([(470, 180), (530, 180), (630, 860), (570, 860)], fill=255)
    white = Image.fromarray(np.where(np.asarray(cut) > 0, 0, np.asarray(white)).astype(np.uint8))
    img.paste(Image.new("RGBA", (S, S), (255, 255, 255, 255)), (0, 0), white)
    return img


def main():
    img = draw()
    OUT.mkdir(parents=True, exist_ok=True)
    img.resize((256, 256), Image.LANCZOS).save(OUT / "broclips.png", optimize=True)
    sizes = [16, 24, 32, 48, 64, 128, 256]
    img.resize((256, 256), Image.LANCZOS).save(OUT / "broclips.ico", sizes=[(s, s) for s in sizes])
    print("wrote", OUT / "broclips.png", "and", OUT / "broclips.ico")


if __name__ == "__main__":
    main()
