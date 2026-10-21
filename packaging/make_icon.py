"""Draw packaging/VNDub.ico (a plain rounded square with the letters VN). Needs Pillow (a dev dependency).

Run by build_app.ps1 when VNDub.ico is missing:  uv run python packaging/make_icon.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).with_name("VNDub.ico")
SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]


def draw(size: int = 256) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    r = size // 5
    d.rounded_rectangle((8, 8, size - 8, size - 8), radius=r, fill=(43, 88, 178, 255))
    d.rounded_rectangle((8, 8, size - 8, size // 2), radius=r, fill=(63, 111, 224, 255))
    try:
        font = ImageFont.load_default(size=int(size * 0.5))
    except TypeError:  # Pillow < 10.1: fixed-size bitmap font
        font = ImageFont.load_default()
    text = "VN"
    box = d.textbbox((0, 0), text, font=font)
    w, h = box[2] - box[0], box[3] - box[1]
    d.text(((size - w) / 2 - box[0], (size - h) / 2 - box[1]), text, font=font, fill=(255, 255, 255, 255))
    return img


if __name__ == "__main__":
    draw().save(OUT, format="ICO", sizes=SIZES)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
