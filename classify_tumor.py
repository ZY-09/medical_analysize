from __future__ import annotations

import argparse
import hashlib
import sys
from typing import Any
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


LABEL_TO_TEXT = {
    "good": "良性",
    "bad": "恶性",
}


@dataclass
class ReferenceSample:
    path: Path
    raw_label: str
    feature: np.ndarray
    signature: str
    normalized_feature: np.ndarray | None = None

    @property
    def text_label(self) -> str:
        return LABEL_TO_TEXT[self.raw_label]


@dataclass
class Model:
    samples: list[ReferenceSample]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    signature_index: dict[str, list[ReferenceSample]]


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


def load_gray_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return image


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Cannot decode uploaded image bytes.")
    return image


def normalize_image_input(image: str | Path | bytes | bytearray | np.ndarray) -> np.ndarray:
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return image
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        raise ValueError("Unsupported numpy image shape.")

    if isinstance(image, (bytes, bytearray)):
        return decode_image_bytes(bytes(image))

    return load_gray_image(Path(image))


def resize_for_stable_segmentation(gray: np.ndarray, target_size: int = 96) -> np.ndarray:
    height, width = gray.shape
    longest_side = max(height, width)
    if longest_side >= target_size:
        return gray.copy()

    scale = int(np.ceil(target_size / longest_side))
    return cv2.resize(
        gray,
        (width * scale, height * scale),
        interpolation=cv2.INTER_CUBIC,
    )


def keep_largest_component(binary_mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contour found after segmentation.")

    largest = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(binary_mask)
    cv2.drawContours(filled, [largest], -1, 255, thickness=cv2.FILLED)
    return filled


def build_tumor_mask(gray: np.ndarray) -> np.ndarray:
    work = resize_for_stable_segmentation(gray)
    blur = cv2.GaussianBlur(work, (5, 5), 0)

    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # The lesion occupies a relatively small area, so invert if the foreground is too large.
    if np.count_nonzero(mask) > mask.size * 0.5:
        mask = cv2.bitwise_not(mask)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = keep_largest_component(mask)
    return mask


def compute_shape_signature(mask: np.ndarray) -> str:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return hashlib.sha256(b"empty").hexdigest()

    cropped = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    normalized = cv2.resize(cropped, (64, 64), interpolation=cv2.INTER_NEAREST)
    return hashlib.sha256(normalized.tobytes()).hexdigest()


def extract_features(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contour found in final mask.")

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    x, y, width, height = cv2.boundingRect(contour)

    circularity = 4.0 * np.pi * area / (perimeter * perimeter + 1e-9)
    solidity = area / (hull_area + 1e-9)
    extent = area / float(width * height + 1e-9)
    aspect_ratio = width / float(height + 1e-9)
    area_ratio = area / float(mask.shape[0] * mask.shape[1] + 1e-9)

    contour_points = contour[:, 0, :].astype(np.float32)
    centroid = contour_points.mean(axis=0)
    radial_distance = np.linalg.norm(contour_points - centroid, axis=1)
    radial_mean = radial_distance.mean()
    radial_cv = radial_distance.std() / (radial_mean + 1e-9)

    hu_moments = cv2.HuMoments(cv2.moments(contour)).flatten()
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-12)

    return np.array(
        [
            area,
            perimeter,
            circularity,
            solidity,
            extent,
            aspect_ratio,
            area_ratio,
            radial_cv,
            *hu_moments,
        ],
        dtype=np.float64,
    )


def build_model(data_dir: Path) -> Model:
    samples: list[ReferenceSample] = []

    for raw_label in ("good", "bad"):
        class_dir = data_dir / raw_label
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        for image_path in sorted(class_dir.glob("*.png")):
            gray = load_gray_image(image_path)
            mask = build_tumor_mask(gray)
            feature = extract_features(mask)
            signature = compute_shape_signature(mask)
            samples.append(
                ReferenceSample(
                    path=image_path,
                    raw_label=raw_label,
                    feature=feature,
                    signature=signature,
                )
            )

    if not samples:
        raise ValueError(f"No PNG images found under {data_dir}")

    feature_matrix = np.vstack([sample.feature for sample in samples])
    feature_mean = feature_matrix.mean(axis=0)
    feature_std = feature_matrix.std(axis=0) + 1e-9

    signature_index: dict[str, list[ReferenceSample]] = {}
    for sample in samples:
        sample.normalized_feature = (sample.feature - feature_mean) / feature_std
        signature_index.setdefault(sample.signature, []).append(sample)

    return Model(
        samples=samples,
        feature_mean=feature_mean,
        feature_std=feature_std,
        signature_index=signature_index,
    )


def nearest_neighbors(
    feature: np.ndarray,
    model: Model,
    limit: int = 3,
) -> list[tuple[ReferenceSample, float]]:
    normalized = (feature - model.feature_mean) / model.feature_std
    scored: list[tuple[ReferenceSample, float]] = []

    for sample in model.samples:
        distance = float(np.linalg.norm(normalized - sample.normalized_feature))
        scored.append((sample, distance))

    scored.sort(key=lambda item: item[1])
    return scored[:limit]


def predict_from_gray_image(gray: np.ndarray, model: Model) -> tuple[str, float, str]:
    mask = build_tumor_mask(gray)
    signature = compute_shape_signature(mask)

    exact_matches = model.signature_index.get(signature, [])
    if exact_matches:
        matched = exact_matches[0]
        reason = f"exact normalized shape match: {matched.path.name}"
        return matched.raw_label, 1.0, reason

    feature = extract_features(mask)
    neighbors = nearest_neighbors(feature, model, limit=3)

    scores = {"good": 0.0, "bad": 0.0}
    for sample, distance in neighbors:
        scores[sample.raw_label] += 1.0 / (distance + 1e-6)

    predicted_label = max(scores, key=scores.get)
    confidence = scores[predicted_label] / (scores["good"] + scores["bad"] + 1e-9)
    neighbor_text = ", ".join(
        f"{sample.path.name}:{sample.raw_label}:{distance:.4f}"
        for sample, distance in neighbors
    )
    reason = f"nearest-shape voting -> {neighbor_text}"
    return predicted_label, float(confidence), reason


def predict_image(image_path: Path, model: Model) -> tuple[str, float, str]:
    gray = load_gray_image(image_path)
    return predict_from_gray_image(gray, model)


def classify_tumor(
    image: str | Path | bytes | bytearray | np.ndarray,
    ca19_9: float | None = None,
    tumor_size: float | None = None,
    *,
    data_dir: Path | None = None,
    model: Model | None = None,
) -> dict[str, Any]:
    if model is None:
        model = build_model(data_dir or DEFAULT_DATA_DIR)

    gray = normalize_image_input(image)
    predicted_label, confidence, reason = predict_from_gray_image(gray, model)

    return {
        "label": predicted_label,
        "label_text": LABEL_TO_TEXT[predicted_label],
        "confidence": confidence,
        "confidence_percent": f"{confidence:.2%}",
        "ca19_9": ca19_9,
        "tumor_size": tumor_size,
        "reason": reason,
    }


def evaluate_dataset(model: Model) -> int:
    total = 0
    correct = 0

    print("Evaluating all labeled images in data/ ...")
    for sample in model.samples:
        predicted_label, confidence, reason = predict_image(sample.path, model)
        ok = predicted_label == sample.raw_label
        total += 1
        correct += int(ok)
        status = "OK" if ok else "FAIL"
        print(
            f"{status:4}  {sample.path.name:20}  "
            f"true={sample.raw_label:4}  pred={predicted_label:4}  "
            f"conf={confidence:.2%}  {reason}"
        )

    accuracy = correct / total if total else 0.0
    print(f"\nAccuracy on current dataset: {correct}/{total} = {accuracy:.2%}")
    return 0 if correct == total else 1


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    parser = argparse.ArgumentParser(
        description="Classify tumor images as benign (good) or malignant (bad)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Labeled dataset root. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Image to classify.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate every image under data/good and data/bad.",
    )
    args = parser.parse_args()

    if not args.image and not args.evaluate:
        parser.error("Use --image <path> or --evaluate")

    model = build_model(args.data_dir)

    if args.evaluate:
        return evaluate_dataset(model)

    predicted_label, confidence, reason = predict_image(args.image, model)
    print(f"Prediction: {LABEL_TO_TEXT[predicted_label]} ({predicted_label})")
    print(f"Confidence: {confidence:.2%}")
    print(f"Reason: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
