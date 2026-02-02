# File: demo_poly/poly.jpg size=278 bytes
# Contains archive/exe marker: True


from __future__ import annotations
from pathlib import Path

ARCHIVE_MARKERS = [b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08", b"Rar!", b"7z\xbc\xaf\x27\x1c", b"MZ"]

def contains_marker(data: bytes, markers: list[bytes] = ARCHIVE_MARKERS) -> bool:
    return any(m in data for m in markers)

def create_polyglot_sample(path: Path) -> None:
    base = b"\xff\xd8\xff" + b"\x00" * 256 + b"\xff\xd9"
    payload = b"\n" + b"PK\x03\x04" + b"FAKEZIPPAYLOAD"
    path.write_bytes(base + payload)

def demo():
    out = Path("demo_poly")
    out.mkdir(exist_ok=True)
    sample = out / "poly.jpg"
    create_polyglot_sample(sample)
    data = sample.read_bytes()
    print(f"File: {sample} size={len(data)} bytes")
    print("Contains archive/exe marker:", contains_marker(data))

if __name__ == "__main__":
    demo()
