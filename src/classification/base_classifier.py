import logging
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Any, Dict, List, Optional, Union
from omegaconf import DictConfig

from src.preprocessing.base_preprocessor import BasePreprocessor
from utils.utils import get_logger

class BaseClassifier(ABC):
    """
    Base Classifier interface
    
    All classifiers must implement the predict method.
    """
    
    def __init__(self, cfg: DictConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize base classifier.
        
        Args:
            cfg: Configuration dictionary
            logger: Optional logger
        """
        self.cfg = cfg
        self.logger = logger or get_logger(__name__)
        self.preprocessor = None
    @abstractmethod
    def predict(self, image: np.ndarray) -> Tuple[int, float]:
        """
        Predict class for an image.
        
        Args:
            image: Image to classify (preprocessed ROI)
            
        Returns:
            Tuple of (class_id, confidence)
        """
        pass
        
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,X_val: Union[np.ndarray,None]=None, y_val: Union[np.ndarray,None]=None) -> Dict[str, Any]:
        """
        Train the classifier with provided data.
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            
        Returns:
            Dictionary with training history or metrics
        """
        pass
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the classifier model to disk.
        
        Args:
            path: Path to save the model
        """
        raise NotImplementedError("Saving not implemented for this classifier")
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the classifier model from disk.
        
        Args:
            path: Path to load the model from
        """
        raise NotImplementedError("Loading not implemented for this classifier")

    def set_preprocessor(self, preprocessor: BasePreprocessor):
        """
        Set the preprocessor for the classifier.
        """
        self.preprocessor = preprocessor

    def get_preprocessor(self) -> BasePreprocessor:
        """
        Get the preprocessor for the classifier.
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not set for the classifier")
        return self.preprocessor
