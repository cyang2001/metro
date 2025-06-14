from __future__ import annotations

# ... existing code ...

from typing import Tuple
import cv2
import numpy as np
from omegaconf import DictConfig

from src.preprocessing.base_preprocessor import BasePreprocessor


class CNNPreprocessor(BasePreprocessor):
    """CNN Preprocessor

    Converts input ROI images to grayscale, resizes them to a fixed size, and normalizes
    pixel values to the range [0, 1]. Optionally applies histogram equalization.

    Attributes
    ----------
    target_size : Tuple[int, int]
        Width and height that every ROI will be resized to.
    equalize_hist : bool
        If True, apply histogram equalization (CLAHE) to improve local contrast.
    """

    def __init__(self, cfg: DictConfig):
        """Initialize the preprocessor with configuration parameters.

        Parameters
        ----------
        cfg : DictConfig
            Configuration containing preprocessing options. Expected keys:
                - target_size: list[int, int], resize target (default: [64, 64])
                - equalize_hist: bool, whether to apply histogram equalization (default: False)
        """
        super().__init__(cfg)
        self.target_size: Tuple[int, int] = tuple(cfg.get("target_size", [64, 64]))  # (w, h)
        self.equalize_hist: bool = cfg.get("equalize_hist", False)

        # CLAHE object is created once to save time when equalize_hist is enabled
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if self.equalize_hist else None


    def preprocess(self, image: np.ndarray) -> np.ndarray:  # type: ignore[override]
        """Preprocess an ROI for CNN inference.

        The function performs the following steps:
        1. Convert the image to 8-bit grayscale if needed.
        2. Resize to the configured target size.
        3. Optionally perform histogram equalization (CLAHE).
        4. Normalize the pixel values into [0, 1] and add a channel dimension.

        Parameters
        ----------
        image : np.ndarray
            Original ROI in BGR or Gray format.

        Returns
        -------
        np.ndarray
            Pre-processed ROI of shape (H, W, 1) with float32 values in [0, 1].
        """
        # 1. Ensure grayscale uint8
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        if gray.dtype != np.uint8:
            # Assume image is float in [0,1] or any other dtype – convert to uint8
            if gray.dtype in (np.float32, np.float64):
                gray = (gray * 255.0).clip(0, 255).astype(np.uint8)
            else:
                gray = gray.astype(np.uint8)

        # 2. Resize (expects (width,height))
        gray = self.resize(gray, self.target_size)

        # 3. Optional histogram equalization (CLAHE)
        if self._clahe is not None:
            gray = self._clahe.apply(gray)

        # 4. Normalize & add channel dim
        gray = gray.astype(np.float32) / 255.0
        gray = np.expand_dims(gray, axis=-1)  # Shape: (H, W, 1)
        return gray