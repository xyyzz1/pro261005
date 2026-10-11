import os
from PIL import Image, ImageDraw, ImageFont

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "tmp")
os.makedirs(OUT, exist_ok=True)

# class names from data.yaml
names = {0: "obj_000001"}

def load_font(size=20):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()

def draw_split(split):
    img_dir = os.path.join(BASE, "images", split)
    lbl_dir = os.path.join(BASE, "labels", split)
    font = load_font(18)
    count = 0
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img_path = os.path.join(img_dir, fname)
        stem = os.path.splitext(fname)[0]
        lbl_path = os.path.join(lbl_dir, stem + ".txt")

        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        draw = ImageDraw.Draw(img)

        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    cls = int(float(parts[0]))
                    cx, cy, w, h = [float(x) for x in parts[1:5]]
                    x1 = int((cx - w / 2) * W)
                    y1 = int((cy - h / 2) * H)
                    x2 = int((cx + w / 2) * W)
                    y2 = int((cy + h / 2) * H)
                    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
                    label = f"{names.get(cls, cls)}"
                    tb = draw.textbbox((x1, y1), label, font=font)
                    draw.rectangle([x1, y1 - (tb[3]-tb[1]+4), tb[2], y1], fill=(255, 0, 0))
                    draw.text((x1, y1 - (tb[3]-tb[1]+2)), label, fill=(255, 255, 255), font=font)
        else:
            draw.text((10, 10), "NO LABEL", fill=(255, 0, 0), font=font)

        out_path = os.path.join(OUT, f"{split}_{fname}")
        img.save(out_path, quality=95)
        count += 1
    print(f"{split}: {count} images -> {OUT}")

draw_split("train")
draw_split("val")
print("done")
