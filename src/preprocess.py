"""
Preprocess module
------------------
grayscale -> CLAHE -> median blur -> Gaussian blur
Each step is saved as an image.
"""

from pathlib import Path
import cv2


def save(path: Path, img) -> None:
    cv2.imwrite(str(path), img)
    print(f"  saved -> {path.name}")


def preprocess(image, output_dir: str):
    """
    image: original BGR image (numpy array)
    Returns the final preprocessed (grayscale, CLAHE-enhanced, denoised) image.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[Preprocess 1] Grayscale")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    save(out / "step1_grayscale.jpg", gray)

    print("[Preprocess 2] CLAHE")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    save(out / "step2_clahe.jpg", enhanced)

    print("[Preprocess 3] Denoise (Median blur)")
    # denoised = cv2.medianBlur(enhanced, 5)
    save(out / "step3_median_blur.jpg", enhanced)

    print("[Preprocess 4] Smooth (Gaussian blur)")
    smoothed = cv2.GaussianBlur(enhanced, (5, 5), 0)
    save(out / "step4_gaussian_blur.jpg", smoothed)

    return smoothed