import os
import cv2
import numpy as np
from src.preprocessing.base_preprocessor import BasePreprocessor
from omegaconf import DictConfig
from utils.utils import get_logger

class ROIParamOptimizerPreprocessor(BasePreprocessor):
    """
    Preprocessor for ROI parameter optimization.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.logger = get_logger(__name__)
        self.use_color_constancy = cfg.get("use_color_constancy", False)
        self.color_constancy_method = cfg.get("color_constancy_method", "grayworld")
        self.use_resize = cfg.get("use_resize", False)
        self.resize_size = cfg.get("resize_size", (64, 64))
        self.debug_visualize = cfg.get("debug_visualize", False)
        self.debug_visualize_dir = cfg.get("debug_visualize_dir")
        self.color_space = "hsv"
    def preprocess(self, image: np.ndarray, has_visualize: bool = True) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("Empty image provided for preprocessing")

        if image.dtype == np.float32 and image.max() <= 1.0:
            img_bgr_uint8 = (image * 255).astype(np.uint8)
        elif image.dtype == np.uint8:
            img_bgr_uint8 = image
        else:
            self.logger.warning("Unexpected image dtype: %s, using uint8", image.dtype)
            img_bgr_uint8 = image.astype(np.uint8)

        original_image_bgr = img_bgr_uint8.copy()

        if self.use_color_constancy:
            img_bgr_uint8 = self._apply_color_constancy(img_bgr_uint8, self.color_constancy_method)

        img_hsv_unit8 = cv2.cvtColor(img_bgr_uint8, cv2.COLOR_BGR2HSV)

        if self.use_resize:
            img_bgr_uint8 = self.resize(img_bgr_uint8, self.resize_size)
            img_hsv_unit8 = self.resize(img_hsv_unit8, self.resize_size)

        if self.debug_visualize and not has_visualize:
            self.logger.info(f"Saving original and processed images to {self.debug_visualize_dir}")
            cv2.imwrite(os.path.join(self.debug_visualize_dir, "original_image_bgr.png"), original_image_bgr)
            cv2.imwrite(os.path.join(self.debug_visualize_dir, "processed_image_bgr.png"), img_bgr_uint8)
            cv2.imwrite(os.path.join(self.debug_visualize_dir, "processed_image_hsv.png"), img_hsv_unit8)

        return img_hsv_unit8
    
    def _apply_color_constancy(self, image, method='ebner', kernel_size=51):
        """
        Apply color constancy to the image.
        
        Args:
            image: Input image
            method: Color constancy method, 'ebner' or 'gray_world'
            kernel_size: Filter size for Ebner method
        
        Returns:
            np.ndarray: Processed image
        """
        if method.lower() == 'ebner':
            return self.ebner_color_constancy(image, kernel_size)
        elif method.lower() == 'gray_world':
            return self.gray_world(image)
        else:
            self.logger.warning(f"Unknown color constancy method: {method}, using original image")
            return image

    def ebner_color_constancy(self, image, kernel_size=51):
        """
        Implement Ebner's color constancy algorithm.
        
        Args:
            image: Input BGR image
            kernel_size: Gaussian filter size
        
        Returns:
            np.ndarray: Processed BGR image
        """
        # Separate channels
        b, g, r = cv2.split(image)
        
        kernel_size = max(3, kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        b_mean = cv2.GaussianBlur(b, (kernel_size, kernel_size), 0)
        g_mean = cv2.GaussianBlur(g, (kernel_size, kernel_size), 0)
        r_mean = cv2.GaussianBlur(r, (kernel_size, kernel_size), 0)
        
        intensity = (b_mean + g_mean + r_mean) / 3.0
        
        intensity[intensity == 0] = 1.0
        
        b_scale = intensity / (b_mean + 1e-6)
        g_scale = intensity / (g_mean + 1e-6)
        r_scale = intensity / (r_mean + 1e-6)
        
        # Apply scaling (prevent excessive scaling)
        max_scale = 3.0  # Prevent excessive amplification
        b_scale = np.clip(b_scale, 0, max_scale)
        g_scale = np.clip(g_scale, 0, max_scale)
        r_scale = np.clip(r_scale, 0, max_scale)
        
        b_corrected = np.clip(b * b_scale, 0, 255).astype(np.uint8)
        g_corrected = np.clip(g * g_scale, 0, 255).astype(np.uint8)
        r_corrected = np.clip(r * r_scale, 0, 255).astype(np.uint8)
        
        result = cv2.merge([b_corrected, g_corrected, r_corrected])
        return result

    def gray_world(self, image):
        """
        Implement Gray World color constancy algorithm.
        
        Args:
            image: Input BGR image
        
        Returns:
            np.ndarray: Processed BGR image
        """
        b, g, r = cv2.split(image)
        
        b_avg = np.mean(b)
        g_avg = np.mean(g)
        r_avg = np.mean(r)
        
        avg = (b_avg + g_avg + r_avg) / 3.0
        
        b_scale = avg / (b_avg + 1e-6)
        g_scale = avg / (g_avg + 1e-6)
        r_scale = avg / (r_avg + 1e-6)
        
        # Apply scaling (prevent excessive scaling)
        max_scale = 3.0  # Prevent excessive amplification
        b_scale = min(b_scale, max_scale)
        g_scale = min(g_scale, max_scale)
        r_scale = min(r_scale, max_scale)
        
        b_corrected = np.clip(b * b_scale, 0, 255).astype(np.uint8)
        g_corrected = np.clip(g * g_scale, 0, 255).astype(np.uint8)
        r_corrected = np.clip(r * r_scale, 0, 255).astype(np.uint8)
        
        result = cv2.merge([b_corrected, g_corrected, r_corrected])
        return result
    def get_color_space(self):
        return self.color_space
