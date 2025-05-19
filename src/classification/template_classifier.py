import os
import cv2
import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional, List, Union
from omegaconf import DictConfig
import glob
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from src.preprocessing.template_preprocessor import TemplatePreprocessor
from src.preprocessing.base_preprocessor import BasePreprocessor
from utils.utils import get_logger, ensure_dir

from src.classification.base_classifier import BaseClassifier
from skimage.metrics import structural_similarity as ssim
class TemplateClassifier(BaseClassifier):
    """
    Template Matching Classifier
    
    Uses OpenCV template matching to classify metro line signs.
    """
    
    def __init__(self, cfg: DictConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize template classifier.
        
        Args:
            cfg: Configuration dictionary
            logger: Optional logger
        """
        super().__init__(cfg, logger)
        self.templates = {}
        self.template_size = tuple(self.cfg.get("template_size", [64, 64]))
        self.template_dir = self.cfg.get("template_dir")
        self.method = eval(self.cfg.get("method"))
        self.threshold = self.cfg.get("threshold")


    def set_preprocessor(self, preprocessor: BasePreprocessor):
        """
        Set the preprocessor for the classifier and initialize templates.
        
        Args:
            preprocessor: Preprocessor to use
        """
        super().set_preprocessor(preprocessor)
        if self.preprocessor is None:
            raise ValueError("Preprocessor is not set")
        if isinstance(self.preprocessor, TemplatePreprocessor):
            self.color_space = self.preprocessor.get_color_space()
        else:
            raise ValueError(f"Preprocessor {self.preprocessor} is not a TemplatePreprocessor")
            


    def _load_templates(self) -> None:

        if not os.path.exists(self.template_dir):
            self.logger.warning(f"Template directory does not exist: {self.template_dir}")
            return
            
        self.logger.info(f"Loading templates from {self.template_dir}")
        
        template_files = []
        for ext in ['png', 'jpg', 'jpeg']:
            template_files.extend(list(Path(self.template_dir).glob(f"*.{ext}")))
        
        if not template_files:
            self.logger.warning("No template files found")
            return
            
        for template_path in template_files:
            try:
                filename = os.path.basename(template_path)
                if not filename.startswith("class_"):
                    continue
                    
                class_id = int(filename.split("_")[1].split(".")[0])
                if self.preprocessor is None:
                    raise ValueError("Preprocessor is not set")
                if self.color_space == "gray":
                    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
                    template = self.preprocessor.resize(template, self.template_size)
                    template = template.astype(np.float32) / 255.0
                else:
                    template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
                    template = self.preprocessor.resize(template, self.template_size)
                    if self.color_space == "rgb":
                        template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
                    elif self.color_space == "lab":
                        template = cv2.cvtColor(template, cv2.COLOR_BGR2LAB)
                    template = template.astype(np.float32)/255.0

                self.templates[class_id] = template

                self.logger.info(f"Loaded template for class {class_id}")
                
            except Exception as e:
                self.logger.error(f"Error loading template {template_path}: {e}")
        
        self.logger.info(f"Loaded {len(self.templates)} templates")
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Create templates from training data.
        
        Args:
            X_train: Training images (preprocessed ROI)
            y_train: Training labels
        """
        self.logger.info("Creating templates from training data")
        
        if len(X_train) != len(y_train):
            self.logger.error(f"Data length mismatch: {len(X_train)} images, {len(y_train)} labels")
            return
            
        ensure_dir(self.template_dir)
        
        class_images = {}
        for i, (image, label) in enumerate(zip(X_train, y_train)):
            if label not in class_images:
                class_images[label] = []
                self.logger.warning(f"Not found class {label} in templates, creating template for it")
            class_images[label].append(image)
        
        for class_id, images in tqdm(class_images.items(), total=len(class_images), 
                                    desc="Creating templates", unit="class"):
            self.logger.info(f"Creating template for class {class_id} from {len(images)} images")
            
            template = np.mean(np.array(images), axis=0)
            
            template_path = os.path.join(self.template_dir, f"class_{class_id}.png")
            template_unit8 = (template * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(template_path, template_unit8)
            
            self.templates[class_id] = template.copy()
            
            self.logger.info(f"Template saved to {template_path} as {self.color_space} color space with dtype {template.dtype}")
        
        self.logger.info(f"Created {len(class_images)} templates")
        
    def predict(self, image: np.ndarray) -> Tuple[int, float]:
        """
        Predict class for a pre-processed (TemplatePreprocessor) ROI.

        Returns
        -------
        class_id : int   (-1 = reject)
        confidence : float in [0,1]
        """
        if not self.templates:
            self._load_templates()
        if not self.templates:
            self.logger.warning("No templates loaded for matching.")
            return -1, 0.0
        if image.shape[:2] != self.template_size:
            self.logger.warning(f"Image size {image.shape[:2]} does not match template size {self.template_size}")
            image = cv2.resize(image, self.template_size)

        if np.var(image) < self.cfg.get("var_threshold", 1e-3):
            return -1, 0.0
        edge_ratio_min = self.cfg.get("edge_ratio_min", 0.05)
        edge_ratio = self._edge_ratio(image)
        if edge_ratio < edge_ratio_min:
            return -1, 0.0

 
        color_space = self.color_space        
        if color_space == "gray":
            best_cls, best_conf = self._match_gray(image)
        elif color_space in ("rgb", "bgr"):
            best_cls, best_conf = self._match_rgb_bgr(image, space=color_space)
        elif color_space == "lab":
            best_cls, best_conf = self._match_lab(image)
        else:
            raise ValueError(f"Unsupported color_space `{color_space}`.")

        if best_conf < self.threshold:
            return -1, best_conf
        return best_cls, best_conf
    
    def save(self, path: str) -> None:
        """
        Save the classifier model to disk.
        
        For template classifier, this is already done in train().
        """
        self.logger.info(f"Templates already saved to {self.template_dir}")
        
    def load(self, path: str) -> None:
        """
        Load the classifier model from disk.
        
        For template classifier, call _load_templates().
        """
        self.logger.info(f"Loading templates from {path}")
        self.template_dir = path
        self._load_templates()
        
    def visualize_templates(self) -> None:
        """
        Visualize all templates.
        """
        if not self.templates:
            self.logger.warning("No templates to visualize")
            return
            
        n_templates = len(self.templates)
        cols = min(5, n_templates)
        rows = (n_templates + cols - 1) // cols
        
        plt.figure(figsize=(cols * 3, rows * 3))
        
        for i, (class_id, template) in enumerate(sorted(self.templates.items())):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(template)
            plt.title(f"Class {class_id}")
            plt.axis('off')
            
        plt.tight_layout()
        plt.show() 
    def _edge_ratio(self, img: np.ndarray) -> float:
        if img.ndim == 3 and self.color_space != "lab":                      
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if self.color_space != "bgr" \
                   else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.ndim == 3 and self.color_space == "lab":
            gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_LAB2RGB), cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        edges = cv2.Canny(gray, 100, 200)
        return float(np.count_nonzero(edges)) / edges.size

    def _match_gray(self, gray: np.ndarray) -> Tuple[int, float]:
        best_cls, best_sim = -1, -1.0
        for cls, tpl in self.templates.items():
            tpl_gray = tpl if tpl.ndim == 2 else cv2.cvtColor(tpl, cv2.COLOR_RGB2GRAY)
            tm_val = cv2.matchTemplate(gray.astype(np.float32),
                                       tpl_gray.astype(np.float32),
                                       self.method)[0, 0]
            ssim_val = ssim(gray, tpl_gray)
            sim = self._combine(tm_val, ssim_val)
            if sim > best_sim:
                best_cls, best_sim = cls, sim
        return best_cls, float(best_sim)

    def _match_rgb_bgr(self, img: np.ndarray, space: str) -> Tuple[int, float]:
        best_cls, best_sim = -1, -1.0
        for cls, tpl in self.templates.items():
            tpl_cur = tpl                                          
            sims = [cv2.matchTemplate(img[:, :, c].astype(np.float32),
                                    tpl_cur[:, :, c].astype(np.float32),
                                    self.method)[0, 0]
                    for c in range(3)]
            tm_val = float(np.mean(sims))
            gray_img = cv2.cvtColor(img,  cv2.COLOR_BGR2GRAY if space == "bgr" else cv2.COLOR_RGB2GRAY)
            gray_tpl = cv2.cvtColor(tpl_cur, cv2.COLOR_BGR2GRAY if space == "bgr" else cv2.COLOR_RGB2GRAY)
            ssim_val = ssim(gray_img, gray_tpl)
            sim = self._combine(tm_val, ssim_val)
            if sim > best_sim:
                best_cls, best_sim = cls, sim
        return best_cls, best_sim


    def _match_lab(self, lab: np.ndarray) -> Tuple[int, float]:
        best_cls, best_sim = -1, -1.0
        for cls, tpl_lab in self.templates.items():
            sims = []
            for c in range(3):          # L, a, b
                sims.append(cv2.matchTemplate(
                    lab[:, :, c].astype(np.float32),
                    tpl_lab[:, :, c].astype(np.float32),
                    self.method
                )[0, 0])
            tm_val = float(np.mean(sims))
            ssim_val = ssim(lab[:, :, 0], tpl_lab[:, :, 0])  # 仅 L 通道
            sim = self._combine(tm_val, ssim_val)
            if sim > best_sim:
                best_cls, best_sim = cls, sim
        return best_cls, float(best_sim)
    def _combine(self, tm_val: float, ssim_val: float) -> float:
        if self.method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
            norm_tm = max(0.0, 1.0 - tm_val)        # 0=完美 →1
        else:
            # 假设 TM_<NORMED> 已在 0~1 之间；如非 *_NORMED，可用 tanh/exp 缩放
            norm_tm = max(0.0, min(1.0, tm_val))
        alpha = self.cfg.get("tm_weight", 0.6)
        return alpha * norm_tm + (1 - alpha) * ssim_val