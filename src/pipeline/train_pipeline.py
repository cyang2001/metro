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
from src.preprocessing.CNN_preprocessor import CNNPreprocessor
from src.classification.CNN_classifier import CNNClassifier
from src.data.roi_dataset import ROIDatasetGenerator, ROIDatasetLoader

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
                cfg=self.cfg.classification.template,
            )

            self.multi_color_detector = MultiColorDetector(
                cfg=self.cfg.roi_detection
            )
            self.multi_color_detector.set_preprocessor(self.roi_param_optimizer_preprocessor)
            self.template_classifier.set_preprocessor(self.template_preprocessor)

            # --- CNN Components ---
            self.cnn_preprocessor = CNNPreprocessor(cfg=self.cfg.preprocessing.cnn)
            self.cnn_classifier = CNNClassifier(cfg=self.cfg.classification.cnn)
            self.cnn_classifier.set_preprocessor(self.cnn_preprocessor)

            self.roi_dataset_generator = ROIDatasetGenerator(cfg=self.cfg.roi_dataset)
            self.roi_dataset_generator.set_preprocessor(self.cnn_preprocessor)

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
        
        # ---------------------------------------------------------------
        # Etape 4: CNN training
        # ---------------------------------------------------------------
        if self.cfg.mode.get("train_cnn", True):
            self.logger.info("Etape 4: CNN training")
            self._train_cnn(train_dataset)
        else:
            self.logger.info("Etape 4: CNN training skipped")

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
        
    # -----------------------------------------------------------------
    def _train_cnn(self, train_dataset: MetroDataset):
        """Generate ROI dataset (if necessary) and train CNN classifier."""
        # 1. Generate ROI dataset if metadata not present
        metadata_path = os.path.join(self.cfg.roi_dataset.get("roi_dataset_dir",
                                os.path.join(self.cfg.output_dir, "roi_dataset")),
                                 "metadata.json")
        try:
            if not os.path.exists(metadata_path):
                self.logger.info("Generating ROI dataset for CNN training…")
                self.roi_dataset_generator.generate_from_dataset(train_dataset)
        except Exception as e:
            self.logger.error(f"Failed to generate ROI dataset: {e}")
            raise

        # 2. Load dataset
        try:
            roi_loader = ROIDatasetLoader(cfg=self.cfg.roi_dataset)
            roi_loader.set_preprocessor(self.cnn_preprocessor)
        except Exception as e:
            self.logger.error(f"Failed to load ROI dataset: {e}")
            raise

        # 3. Split into train/val
        X_train, y_train, X_val, y_val = roi_loader.split_train_val(
            val_ratio=self.cfg.get("cnn_val_ratio", 0.2),
            random_seed=self.cfg.seed,
        )
        if len(X_train) == 0:
            self.logger.error("Empty ROI training set – cannot train CNN.")
            return

        # 4. Compute class weights to handle imbalance (indices 0-13)
        class_weights_raw = roi_loader.get_class_balance_weights()
        # Map class_id → idx
        class_weights = {
            self.cnn_classifier.label_to_index[cls]: weight
            for cls, weight in class_weights_raw.items() if cls in self.cnn_classifier.label_to_index
        }

        # 5. Train CNN classifier
        self.logger.info("Training CNN classifier…")
        history = self.cnn_classifier.train(
            X_train, y_train,
            X_val=X_val if len(X_val) > 0 else None,
            y_val=y_val if len(y_val) > 0 else None,
            class_weight=class_weights or None,
        )

        # 6. Plot training history if configured
        if history and self.cfg.mode.get("plot_cnn_history", False):
            plot_training_history(history, title="CNN Training History")

def main(cfg: DictConfig):
    logger = get_logger(__name__)
    try:
        pipeline = MetroTrainPipeline(cfg, logger)
        pipeline.run()
        logger.info("Training pipeline completed successfully")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

