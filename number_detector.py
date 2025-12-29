"""number_detector.py

SUB ai - Number Detection

This module started as a simple single-digit MNIST classifier wrapper.
In this round we significantly upgraded the vision pipeline to support:

* Robust preprocessing (thresholding + noise cleanup)
* Digit segmentation (multi-digit images)
* MNIST-like normalization for each segmented digit
* Optional model-based classification when a trained model is available

The public API remains compatible:
    detector = NumberDetector(model_path='models/sub_ai_model_latest.h5')
    result = detector.detect('path/to/image.png')

The returned dict now includes extra keys for multi-digit predictions:
    predicted_number, predicted_digits, digit_confidences, boxes
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import os

import cv2
import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    """Simple bounding box for a detected digit region."""

    x: int
    y: int
    w: int
    h: int

    def as_dict(self) -> Dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


class NumberDetector:
    """Detect and recognize digits (0-9) from images.

    If a trained MNIST digit classifier is available, we run per-digit
    classification. If not, we still perform segmentation-based detection
    (useful for demos and basic "number vs not" heuristics).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        *,
        min_confidence: float = 0.70,
        max_digits: int = 12,
    ):
        """Initialize the detector.

        Args:
            model_path: Path to a saved Keras model (optional).
            min_confidence: Minimum per-digit confidence to mark the image as a number.
            max_digits: Safety limit to avoid trying to decode extremely noisy images.
        """

        self.model = None
        self.model_path = model_path
        self.min_confidence = float(min_confidence)
        self.max_digits = int(max_digits)

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("No pre-trained model loaded. Using segmentation-based detection.")

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess an image into a single 28x28 normalized array.

        This method is kept for backwards compatibility with older code.
        For multi-digit images, the first detected digit region is returned.
        """

        gray = self._load_grayscale(image_path)
        boxes, rois = self._segment_digit_rois(gray)

        if rois:
            return self._roi_to_mnist(rois[0])

        # Fallback to a simple resize if segmentation fails.
        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        return (resized.astype(np.float32) / 255.0)

    def detect(self, image_path: str) -> Dict[str, Any]:
        """Detect whether an image contains digits and optionally recognize them.

        Returns:
            A dictionary compatible with previous versions, with extra keys:
                predicted_number: str | None
                predicted_digits: list[int] | None
                digit_confidences: list[float] | None
                boxes: list[dict]
        """

        try:
            gray = self._load_grayscale(image_path)
            boxes, rois = self._segment_digit_rois(gray)

            # If segmentation is clearly unusable, mark as not a number.
            if not self._segmentation_looks_like_digits(gray, boxes):
                return {
                    "status": "success",
                    "is_number": False,
                    "predicted_digit": None,
                    "predicted_digits": None,
                    "predicted_number": None,
                    "digit_confidences": None,
                    "boxes": [b.as_dict() for b in boxes],
                    "message": "This does not look like a number image.",
                    "confidence": 0.0,
                    "method": "segmentation_heuristic" if self.model is None else "neural_network",
                }

            if self.model is None:
                # Segmentation-only mode (no classification available)
                return {
                    "status": "success",
                    "is_number": True,
                    "predicted_digit": None,
                    "predicted_digits": None,
                    "predicted_number": None,
                    "digit_confidences": None,
                    "boxes": [b.as_dict() for b in boxes],
                    "message": f"Found {len(boxes)} digit-like region(s). (Model not loaded)",
                    "confidence": 0.5,
                    "method": "segmentation_heuristic",
                }

            # Model-based classification (single or multi-digit)
            predictions: List[int] = []
            confidences: List[float] = []

            for roi in rois:
                mnist_digit = self._roi_to_mnist(roi)
                digit, conf = self._predict_digit(mnist_digit)
                predictions.append(digit)
                confidences.append(conf)

            predicted_number = "".join(str(d) for d in predictions) if predictions else None
            overall_confidence = float(np.min(confidences)) if confidences else 0.0

            is_number = bool(predictions) and overall_confidence >= self.min_confidence

            if not predictions:
                message = "This does not look like a number image."
            elif is_number:
                message = (
                    f"This is a number image! Detected: {predicted_number}"
                    if len(predictions) > 1
                    else f"This is a number image! Detected: {predictions[0]}"
                )
            else:
                message = (
                    f"I found possible digits '{predicted_number}', but I'm not confident."
                    if predicted_number
                    else "I couldn't confidently recognize a digit."
                )

            return {
                "status": "success",
                "is_number": is_number,
                "predicted_digit": predictions[0] if len(predictions) == 1 and is_number else None,
                "predicted_digits": predictions if predictions else None,
                "predicted_number": predicted_number if predictions else None,
                "digit_confidences": confidences if confidences else None,
                "boxes": [b.as_dict() for b in boxes],
                "message": message,
                "confidence": overall_confidence,
                "method": "neural_network",
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "is_number": False,
                "predicted_digit": None,
                "predicted_digits": None,
                "predicted_number": None,
                "digit_confidences": None,
                "boxes": [],
                "confidence": 0.0,
            }

    # ---------------------------------------------------------------------
    # Model I/O
    # ---------------------------------------------------------------------
    def load_model(self, model_path: str) -> None:
        """Load a saved Keras model."""

        try:
            import tensorflow as tf  # noqa: F401

            from tensorflow import keras

            self.model = keras.models.load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def save_model(self, save_path: str) -> None:
        """Save the current model."""

        if self.model is not None:
            self.model.save(save_path)
            print(f"Model saved to {save_path}")
        else:
            print("No model to save.")

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _load_grayscale(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        return img

    def _binarize(self, gray: np.ndarray) -> np.ndarray:
        """Return a binary image with *foreground digits as white* (255)."""

        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Heuristic inversion: if image background is light, invert so digits become white.
        use_inv = float(np.mean(blurred)) > 127.0
        thresh_flag = cv2.THRESH_BINARY_INV if use_inv else cv2.THRESH_BINARY
        _, binary = cv2.threshold(blurred, 0, 255, thresh_flag + cv2.THRESH_OTSU)

        # Reduce small noise while keeping stroke integrity.
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        return binary

    def _segment_digit_rois(self, gray: np.ndarray) -> Tuple[List[BoundingBox], List[np.ndarray]]:
        """Segment digits from an image.

        Returns:
            (boxes, rois) where each ROI is a binary sub-image (digits white on black).
        """

        binary = self._binarize(gray)
        contours, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h_img, w_img = gray.shape[:2]
        img_area = float(h_img * w_img)

        # Dynamic thresholds based on image size.
        min_area = max(25.0, img_area * 0.0005)  # 0.05% of the image
        min_h = max(8, int(h_img * 0.25))

        boxes: List[BoundingBox] = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = float(w * h)
            if area < min_area:
                continue
            if h < min_h:
                continue

            # Filter extremely flat / wide noise.
            aspect = w / float(h + 1e-6)
            if aspect > 2.5 or aspect < 0.08:
                continue

            boxes.append(BoundingBox(int(x), int(y), int(w), int(h)))

        # Sort left-to-right for multi-digit numbers.
        boxes.sort(key=lambda b: b.x)

        # Limit the number of decoded digits to avoid noise explosions.
        if len(boxes) > self.max_digits:
            # Keep the largest boxes by area as a best effort.
            boxes = sorted(boxes, key=lambda b: b.w * b.h, reverse=True)[: self.max_digits]
            boxes.sort(key=lambda b: b.x)

        rois: List[np.ndarray] = []
        for b in boxes:
            # Add padding around the digit.
            pad = int(0.20 * max(b.w, b.h))
            x1 = max(b.x - pad, 0)
            y1 = max(b.y - pad, 0)
            x2 = min(b.x + b.w + pad, w_img)
            y2 = min(b.y + b.h + pad, h_img)
            roi = binary[y1:y2, x1:x2]
            rois.append(roi)

        return boxes, rois

    def _segmentation_looks_like_digits(self, gray: np.ndarray, boxes: List[BoundingBox]) -> bool:
        """Heuristic gate to reduce false positives on noisy images."""

        if not boxes:
            return False

        h_img, w_img = gray.shape[:2]
        img_area = float(h_img * w_img)

        # If there are *too many* regions, it's almost always noise.
        if len(boxes) > self.max_digits:
            return False

        # Total area of boxes should not cover the whole image (noise) nor be too tiny.
        total_box_area = float(sum(b.w * b.h for b in boxes))
        area_ratio = total_box_area / (img_area + 1e-6)

        if area_ratio < 0.01:  # too little foreground
            return False
        if area_ratio > 0.90:  # boxes cover almost everything
            return False

        return True

    def _roi_to_mnist(self, roi_binary: np.ndarray) -> np.ndarray:
        """Convert a binary ROI (digits white on black) into MNIST-like 28x28 float."""

        if roi_binary.ndim != 2:
            raise ValueError("ROI must be a single-channel (grayscale/binary) image")

        # Crop to the tight bounding box of foreground pixels.
        coords = cv2.findNonZero(roi_binary)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            roi = roi_binary[y : y + h, x : x + w]
        else:
            roi = roi_binary

        # Guard against empty crop.
        if roi.size == 0:
            return np.zeros((28, 28), dtype=np.float32)

        # Resize the digit to fit a 20x20 box while preserving aspect ratio.
        h, w = roi.shape[:2]
        scale = 20.0 / float(max(h, w))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Place into a 28x28 canvas.
        canvas = np.zeros((28, 28), dtype=np.uint8)
        x_off = (28 - new_w) // 2
        y_off = (28 - new_h) // 2
        canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized

        return canvas.astype(np.float32) / 255.0

    def _predict_digit(self, mnist_digit: np.ndarray) -> Tuple[int, float]:
        """Predict a digit and confidence from a 28x28 normalized input."""

        if self.model is None:
            raise RuntimeError("Model is not loaded")

        img_input = mnist_digit.reshape(1, 28, 28, 1)
        probs = self.model.predict(img_input, verbose=0)[0]
        digit = int(np.argmax(probs))
        conf = float(np.max(probs))
        return digit, conf


if __name__ == "__main__":
    detector = NumberDetector()

    print("SUB ai - Number Detector")
    print("=" * 50)
    print("Usage: detector.detect('path/to/image.jpg')")
    print("\nReady for detection!")
