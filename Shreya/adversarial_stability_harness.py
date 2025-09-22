
from __future__ import annotations
import io
from pathlib import Path
from PIL import Image

def default_classifier(image_bytes: bytes) -> tuple[str, float]:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return ("invalid", 0.0)
    r = g = b = 0
    for pr, pg, pb in img.getdata():
        r += pr; g += pg; b += pb
    tot = r + g + b + 1e-9
    maxc = max(r, g, b)
    conf = float(maxc / tot)
    if maxc == r: return ("red", conf)
    if maxc == g: return ("green", conf)
    return ("blue", conf)

def small_transforms(img: Image.Image) -> list[Image.Image]:
    transforms = [img]
    b = io.BytesIO()
    img.save(b, format="JPEG", quality=85)
    transforms.append(Image.open(io.BytesIO(b.getvalue())))
    transforms.append(img.resize((max(1, img.width - 1), max(1, img.height - 1))))
    transforms.append(img.rotate(1))
    return transforms

def stability_check(image_bytes: bytes, classifier_fn=default_classifier) -> dict:
    try:
        base_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return {"error": "cannot_open", "unstable": True}
    transforms = small_transforms(base_img)
    orig_label, orig_conf = classifier_fn(image_bytes)
    mismatches = []
    for t in transforms[1:]:
        b = io.BytesIO()
        t.save(b, format="JPEG", quality=90)
        label, conf = classifier_fn(b.getvalue())
        if label != orig_label:
            mismatches.append((label, conf))
    unstable = bool(mismatches)
    return {"original": (orig_label, orig_conf), "mismatches": mismatches, "unstable": unstable}

def demo():
    out = Path("demo_adv")
    out.mkdir(exist_ok=True)
    img_path = out / "sample.jpg"
    Image.new("RGB", (240, 160), color=(200, 50, 60)).save(img_path)
    data = img_path.read_bytes()
    res = stability_check(data, default_classifier)
    print("Stability check result:", res)

if __name__ == "__main__":
    demo()
