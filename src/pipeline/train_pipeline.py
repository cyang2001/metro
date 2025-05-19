"""
Author: @Chen YANG
"""
import logging
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union, Callable
from omegaconf import DictConfig
import tensorflow as tf
import keras
import keras_tuner as kt
import cv2
from tqdm import tqdm

from src.roi_detection.multi_color_detector import MultiColorDetector
from src.preprocessing.template_preprocessor import TemplatePreprocessor
from src.preprocessing.roi_preprocessor import ROIParamOptimizerPreprocessor
from src.classification.template_classifier import TemplateClassifier
from src.data.dataset import MetroDataset
from utils.utils import get_logger, ensure_dir, plot_training_history, save_confusion_matrix

class MetroTrainPipeline:
    """
    Paris Metro Line Recognition Training Pipeline
    
    Manages the training process including:
    1. Dataset loading and preprocessing
    2. Template creation
    3. CNN model training and evaluation but not work
    """
    
    def __init__(self, cfg: DictConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize the training pipeline.
        
        Args:
            cfg: Configuration object
            logger: Optional logger instance
        """
        if logger is None:
            self.logger = get_logger(__name__)
        else:
            self.logger = logger
        self.cfg = cfg
        self._init_components()
    
    def _init_components(self):
        """
        Initialize pipeline components.
        """
        try:
            self.logger.info("Initializing training components...")
            
            self.template_preprocessor = TemplatePreprocessor(
                cfg=self.cfg.preprocessing.template
            )

            self.roi_param_optimizer_preprocessor = ROIParamOptimizerPreprocessor(
                cfg=self.cfg.preprocessing.roi_param_optimizer
            )

            self.template_classifier = TemplateClassifier(
                cfg=self.cfg.classification,
            )

            self.multi_color_detector = MultiColorDetector(
                cfg=self.cfg.roi_detection
            )
            self.multi_color_detector.set_preprocessor(self.roi_param_optimizer_preprocessor)
            self.template_classifier.set_preprocessor(self.template_preprocessor)

        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def run(self):
        """
        Run the training pipeline.
        """
        self.logger.info("=== Start Training Pipeline ===")
        self.logger.info("Etape 1: Loading dataset")
        train_dataset, val_dataset = self._load_datasets()

        if self.cfg.mode.get("optimize_roi_param", False):
            self.logger.info("Etape 2: Optimizing ROI parameters")
            self._optimize_roi_param(train_dataset)
        else:
            self.logger.info("Etape 2: Optimizing ROI parameters skipped")
        
        if self.cfg.mode.get("create_templates", False):
            self.logger.info("Etape 3: Creating templates")
            self._create_templates(train_dataset)
        else:
            self.logger.info("Etape 3: Creating templates skipped")
        
    def _load_datasets(self) -> Tuple[MetroDataset, MetroDataset]:
        """
        Load and prepare training and validation datasets.
        
        Returns:
            Tuple of (training dataset, validation dataset)
        """
        self.logger.info("Loading datasets...")
        
        # Load training dataset
        train_dataset = MetroDataset(
            cfg=self.cfg.dataset,
            mode='train',
        )
        
        # Load validation dataset
        val_dataset = MetroDataset(
            cfg=self.cfg.dataset,
            mode='val',
        )
        
        self.logger.info(f"Datasets loaded: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
        
        return train_dataset, val_dataset

    def _optimize_roi_param(self, train_dataset: MetroDataset):
        """
        Optimize ROI parameters.
        """
        self.logger.info("Optimizing ROI parameters...")
        self.multi_color_detector.optimize_color_parameters(train_dataset, visualize=False)

    def _create_templates(self, train_dataset: MetroDataset):
        """
        Create templates.
        """
        self.logger.info("Creating templates...")

        X_train_raw, y_train = train_dataset.get_all()

        if len(X_train_raw) == 0:
            self.logger.error("No training data found")
            return
        
        if len(y_train) == 0:
            self.logger.error("No training labels found")
            return
            
        original_shapes = []
        processed_images = []
        has_visualize = False
        for i, img in tqdm(enumerate(X_train_raw), total=len(X_train_raw), 
                          desc="Processing images", unit="img"):
            original_shapes.append(img.shape)

            # Visualize the image randomly
            random_int = np.random.randint(0, 1)
            if random_int == 0 and not has_visualize:
                has_visualize = True
            
            preprocessed_img = self.template_preprocessor.preprocess(img, has_visualize)
            processed_images.append(preprocessed_img)
        
        processed_images = np.array(processed_images)
        self.template_classifier.train(processed_images, y_train)

        self.logger.info("Templates created successfully")
        

def main(cfg: DictConfig):
    logger = get_logger(__name__)
    try:
        pipeline = MetroTrainPipeline(cfg, logger)
        pipeline.run()
        logger.info("Training pipeline completed successfully")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

