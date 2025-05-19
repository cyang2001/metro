
from abc import ABC, abstractmethod
from typing import Any, Tuple
import cv2
import numpy as np
from omegaconf import DictConfig


class BasePreprocessor(ABC):
    """
    Base class for all preprocessors.
    """
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
    
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the data.
        """
        return image

    def resize(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize the image to the target size.
        """
        return cv2.resize(image, target_size)
