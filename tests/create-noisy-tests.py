from pathlib import Path
import random

import cv2
import numpy as np

FOLDER = Path(r"C:\Users\T.Abdellahi\Desktop\term8\vision\proj\FINAL\Sudoku-Solver\data\bad-sodu")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

random.seed()


def add_salt_pepper(img):
    out = img.copy()

    h, w = out.shape[:2]
    amount = 0.08

    n = int(h * w * amount)

    ys = np.random.randint(0, h, n)
    xs = np.random.randint(0, w, n)
    out[ys, xs] = 255

    ys = np.random.randint(0, h, n)
    xs = np.random.randint(0, w, n)
    out[ys, xs] = 0

    return out


def add_gaussian_noise(img):
    noise = np.random.normal(0, 40, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def rotate(img):
    angle = random.uniform(-30, 30)

    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    return cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderValue=255,
    )


def add_shadow(img):
    out = img.astype(np.float32)

    h, w = out.shape[:2]

    side = random.choice(["left", "right", "top", "bottom"])

    if side == "left":
        mask = np.tile(np.linspace(0.25, 1.0, w), (h, 1))
    elif side == "right":
        mask = np.tile(np.linspace(1.0, 0.25, w), (h, 1))
    elif side == "top":
        mask = np.tile(np.linspace(0.25, 1.0, h), (w, 1)).T
    else:
        mask = np.tile(np.linspace(1.0, 0.25, h), (w, 1)).T

    out *= mask

    return np.clip(out, 0, 255).astype(np.uint8)


OPERATIONS = [
    ("sp", add_salt_pepper),
    ("gauss", add_gaussian_noise),
    ("rot", rotate),
    ("shadow", add_shadow),
]


for path in FOLDER.iterdir():
    if path.suffix.lower() not in IMAGE_EXTS:
        continue

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    out = img.copy()

    selected = random.sample(OPERATIONS, 2)

    suffix = []

    for name, func in selected:
        out = func(out)
        suffix.append(name)

    new_name = f"{path.stem}_{'_'.join(suffix)}{path.suffix}"

    cv2.imwrite(str(path.parent / new_name), out)

    print(f"Saved: {new_name}")

print("Done.")