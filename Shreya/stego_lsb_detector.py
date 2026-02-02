#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from PIL import Image

def lsb_fraction_red_channel(img: Image.Image, sample_limit: int = 100_000) -> float:
    img = img.convert("RGB")
    pixels = list(img.getdata())
    total = len(pixels)
    step = max(1, total // sample_limit)
    ones, count = 0, 0
    for i in range(0, total, step):
        r, g, b = pixels[i]
        ones += (r & 1)
        count += 1
    return ones / count if count else 0.0

def lsb_entropy_score(img: Image.Image) -> float:
    p = lsb_fraction_red_channel(img)
    ent = p * (1 - p)
    return ent / 0.25

def demo_create_stego_and_check(outdir: Path):
    outdir.mkdir(exist_ok=True)
    clean = outdir / "clean.png"
    img_clean = Image.new("RGB", (300, 300), color=(120, 120, 200))
    img_clean.save(clean)

    stego = outdir / "stego.png"
    img = Image.new("RGB", (300, 300))
    pixels = []
    toggle = 0
    for y in range(300):
        for x in range(300):
            r = 120 | (toggle & 1)
            pixels.append((r, 120, 120))
            toggle ^= 1
    img.putdata(pixels)
    img.save(stego)

    for p in (clean, stego):
        img = Image.open(p)
        frac = lsb_fraction_red_channel(img)
        score = lsb_entropy_score(img)
        print(f"{p.name}: LSB_ones_frac={frac:.4f}, entropy_score={score:.4f}")

if __name__ == "__main__":
    demo_create_stego_and_check(Path("demo_stego"))
