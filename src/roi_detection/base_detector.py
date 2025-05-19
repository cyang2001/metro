"""
Base classes and factory functions for ROI detection.
"""

import json
import logging
from abc import ABC, abstractmethod
import os
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from omegaconf import DictConfig

from src.preprocessing.base_preprocessor import BasePreprocessor
from utils.utils import get_logger

class BaseDetector(ABC):
    """
    Base class for ROI detectors.
    
    All detectors must implement the detect method.
    """
    
    def __init__(self, cfg: DictConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize the detector.
        
        Args:
            cfg: Configuration dictionary
            logger: Optional logger
        """
        self.cfg = cfg
        self.logger = logger or get_logger(__name__)
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect ROIs in the image.
        
        Args:
            image: Input image
            
        Returns:
            List of detected ROIs as Dicts with keys:
            - 'bbox': List[int] (x1, y1, x2, y2)
            - 'line_id': str
            - 'confidence': float
        """
        pass
    
    def update_params(self, params: Dict[str, Any]) -> None:
        """
        Update detector parameters.
        
        Args:
            params: Parameter dictionary
        """
        self.logger.warning("update_params not implemented in this detector")
    @abstractmethod
    def save_params(self, params: Dict[str, Any]) -> bool:
        """
        Save detector parameters.
        
        Args:
            params_dir: Directory to save parameters
        """
        pass

    def set_preprocessor(self, preprocessor: BasePreprocessor):
        self.preprocessor = preprocessor
    
    def get_preprocessor(self) -> BasePreprocessor:
        return self.preprocessor
