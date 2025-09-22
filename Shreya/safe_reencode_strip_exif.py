# Original: demo_normalize/orig_sample.jpg size=2075
# Normalized: demo_normalize/normalized.jpg size=2075
# Extracted EXIF keys: []

from __future__ import annotations
import io
from pathlib import Path
from PIL import Image, ExifTags, UnidentifiedImageError

def extract_exif_map(img: Image.Image) -> dict:
    out = {}
    try:
        exif = img.getexif()
        if not exif:
            return out
        for k, v in exif.items():
            tag = ExifTags.TAGS.get(k, k)
            out[str(tag)] = str(v)
    except Exception:
        pass
    return out

def normalize_image_bytes(input_bytes: bytes, max_size=(2048, 2048), quality=90) -> tuple[bytes, dict]:
    with Image.open(io.BytesIO(input_bytes)) as im:
        exif_map = extract_exif_map(im)
        im = im.convert("RGB")
        im.thumbnail(max_size)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality)
        return buf.getvalue(), exif_map

def demo_create_and_normalize(outdir: Path):
    outdir.mkdir(exist_ok=True)
    src = outdir / "orig_sample.jpg"
    im = Image.new("RGB", (400, 300), color=(123, 200, 100))
    im.save(src, format="JPEG", quality=90)
    data = src.read_bytes()
    norm_bytes, exif = normalize_image_bytes(data)
    norm_path = outdir / "normalized.jpg"
    norm_path.write_bytes(norm_bytes)
    print("Original:", src, "size=", src.stat().st_size)
    print("Normalized:", norm_path, "size=", norm_path.stat().st_size)
    print("Extracted EXIF keys:", list(exif.keys()))

if __name__ == "__main__":
    demo_create_and_normalize(Path("demo_normalize"))
