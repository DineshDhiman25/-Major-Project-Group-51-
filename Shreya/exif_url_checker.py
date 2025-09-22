
from __future__ import annotations
from pathlib import Path
from PIL import Image, ExifTags

def extract_exif_text(img_path: Path) -> dict:
    out = {}
    with Image.open(img_path) as img:
        exif = img.getexif()
        if not exif:
            return out
        for k, v in exif.items():
            tag = ExifTags.TAGS.get(k, str(k))
            try:
                out[str(tag)] = str(v)
            except Exception:
                out[str(tag)] = repr(v)
    return out

def exif_contains_url(exif_map: dict) -> bool:
    for v in exif_map.values():
        if "http://" in v.lower() or "https://" in v.lower():
            return True
    return False

def demo():
    out = Path("demo_exif")
    out.mkdir(exist_ok=True)
    sample = out / "img.jpg"
    img = Image.new("RGB", (200, 200), color=(100, 150, 200))
    img.save(sample)
    ex = extract_exif_text(sample)
    print("EXIF map keys:", list(ex.keys()))
    print("Contains http(s):", exif_contains_url(ex))

if __name__ == "__main__":
    demo()
