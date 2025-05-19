"""
Author: @Chen YANG
"""
import os
import cv2
import numpy as np
from src.preprocessing.base_preprocessor import BasePreprocessor
from omegaconf import DictConfig
from utils.utils import get_logger
class TemplatePreprocessor(BasePreprocessor):
    """
    Preprocessor for template matching.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.logger = get_logger(__name__)
        self.use_clache = cfg.get("use_clache", False)
        self.use_blur = cfg.get("use_blur", False)
        self.use_resize = cfg.get("use_resize", False)
        self.resize_size = cfg.get("resize_size", (64, 64))
        self.debug_visualize = cfg.get("debug_visualize", False)
        self.debug_visualize_dir = cfg.get("debug_visualize_dir")
        self.color_space = cfg.get("color_space", "rgb")
    def preprocess(self, image: np.ndarray, has_visualize: bool = False) -> np.ndarray:
        if self.debug_visualize and not has_visualize:
            self.logger.info(f"Applying preprocessing")
            original_image = image.copy()

        if self.color_space == "gray":
            gray_image = cv2.cvtColor((image * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
            image = gray_image.astype(np.float32) / 255.0
            
            if self.use_clache:
                uint8_image = (image * 255).clip(0, 255).astype(np.uint8)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                image = clahe.apply(uint8_image).astype(np.float32) / 255.0
        
        elif self.color_space == "rgb":
            rgb_image = cv2.cvtColor((image * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            image = rgb_image.astype(np.float32) / 255.0
            if self.use_clache:
                uint8_image = (image * 255).clip(0, 255).astype(np.uint8)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                channels = list(cv2.split(uint8_image))  
                for i in range(len(channels)):
                    channels[i] = clahe.apply(channels[i])

                uint8_image = cv2.merge(channels)
                image = uint8_image.astype(np.float32) / 255.0
        
        elif self.color_space == "bgr":

            if self.use_clache:
                uint8_image = (image * 255).clip(0, 255).astype(np.uint8)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                channels = list(cv2.split(uint8_image)) 
                for i in range(len(channels)):
                    channels[i] = clahe.apply(channels[i])
                uint8_image = cv2.merge(channels)
                image = uint8_image.astype(np.float32) / 255.0
        
        elif self.color_space == 'lab':
            lab_image = cv2.cvtColor((image * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
            image = lab_image.astype(np.float32) / 255.0
            
            if self.use_clache:
                uint8_image = (image * 255).clip(0, 255).astype(np.uint8)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                channels = list(cv2.split(uint8_image))  
                channels[0] = clahe.apply(channels[0])
                uint8_image = cv2.merge(channels)
                image = uint8_image.astype(np.float32) / 255.0

        if self.use_blur:
            image = cv2.GaussianBlur(image, (3, 3), 0)
        
        if self.debug_visualize and not has_visualize:
            self.logger.info(f"Saving original and processed images to {self.debug_visualize_dir}")
            cv2.imwrite(os.path.join(self.debug_visualize_dir, "original_image.png"), (original_image * 255).clip(0, 255).astype(np.uint8))# type: ignore
            cv2.imwrite(os.path.join(self.debug_visualize_dir, "processed_image.png"), (image * 255).clip(0, 255).astype(np.uint8))

        if self.use_resize:
            image = self.resize(image, self.resize_size)

        return image

    def get_color_space(self):
        return self.color_space
