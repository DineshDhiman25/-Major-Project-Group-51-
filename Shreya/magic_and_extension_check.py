# sample.jpg -> magic: image/jpeg, double_ext: False, size: 104 bytes
# photo.jpg.exe -> magic: None, double_ext: True, size: 202 bytes


from __future__ import annotations
import binascii
from pathlib import Path

MAGIC_SIGNATURES = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"GIF87a": "image/gif",
    b"GIF89a": "image/gif",
    b"%PDF-": "application/pdf",
    b"<?xml": "xml_like",
}

def detect_magic(data: bytes) -> str | None:
    head = data[:16]
    for sig, mime in MAGIC_SIGNATURES.items():
        if head.startswith(sig):
            return mime
    if b"<svg" in data[:512].lower():
        return "image/svg+xml"
    return None

def has_double_extension(filename: str) -> bool:
    parts = filename.split(".")
    if len(parts) < 3:
        return False
    last = parts[-1].lower()
    image_exts = {"jpg", "jpeg", "png", "gif", "bmp", "webp"}
    return last not in image_exts

def demo_write_sample_files(outdir: Path) -> tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    jpeg = outdir / "sample.jpg"
    with open(jpeg, "wb") as f:
        f.write(b"\xff\xd8\xff" + b"\x00" * 100 + b"\xff\xd9")
    fake = outdir / "photo.jpg.exe"
    with open(fake, "wb") as f:
        f.write(b"MZ" + b"\x00" * 200)
    return jpeg, fake

def run_demo():
    d = Path("demo_magic")
    j, f = demo_write_sample_files(d)
    for p in (j, f):
        data = p.read_bytes()
        magic = detect_magic(data)
        print(f"{p.name} -> magic: {magic}, double_ext: {has_double_extension(p.name)}, size: {len(data)} bytes")

if __name__ == "__main__":
    run_demo()
