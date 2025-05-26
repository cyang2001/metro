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

# 新增导入
from src.roi_detection.roi_dataset_generator import ROIDatasetGenerator
from src.data.roi_dataset_loader import ROIDatasetLoader

class MetroTrainPipeline:
    """
    Paris Metro Line Recognition Training Pipeline
    
    Manages the training process including:
    1. Dataset loading and preprocessing
    2. Template creation
    3. CNN model training and evaluation
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

            # 新增：可选CNN预处理器
            self.cnn_preprocessor = getattr(self, "cnn_preprocessor", None)
            # 新增：ROI数据集配置
            self.roi_dataset_cfg = self.cfg.get("roi_dataset_cfg", {})

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

        # 新增：训练CNN
        if self.cfg.mode.get("train_cnn", False):
            self.logger.info("Etape 4: Training CNN")
            self._train_cnn(train_dataset, val_dataset)
        else:
            self.logger.info("Etape 4: Training CNN skipped")
        
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
    
    def _train_cnn(self, train_dataset: MetroDataset, val_dataset: MetroDataset) -> Dict[str, Any]:
        """
        Train CNN classifier using ROIs extracted from dataset.
        """
        self.logger.info("Training CNN classifier using ROI-based approach...")

        roi_dataset_config = self.roi_dataset_cfg
        train_roi_dir = os.path.join(roi_dataset_config.get("train_dir", "data/train"))
        val_roi_dir = os.path.join(roi_dataset_config.get("val_dir", "data/val"))

        ensure_dir(train_roi_dir)
        ensure_dir(val_roi_dir)

        self.logger.info("Generating ROI datasets for training...")

        cnn_preprocessor = None
        if hasattr(self.template_classifier, 'get_preprocessor'):
            cnn_preprocessor = self.template_classifier.get_preprocessor()
        else:
            cnn_preprocessor = self.cnn_preprocessor

        train_roi_generator = ROIDatasetGenerator(
            cfg=roi_dataset_config,
        )

        train_roi_generator.output_dir = train_roi_dir
        train_roi_generator.roi_dir = os.path.join(train_roi_dir, "rois")
        train_roi_generator.metadata_path = os.path.join(train_roi_dir, "metadata.json")

        ensure_dir(train_roi_generator.roi_dir)

        train_metadata = train_roi_generator.generate_from_dataset(train_dataset)
        self.logger.info(f"Generated training ROI dataset with {train_metadata['total_samples']} samples")

        val_roi_generator = ROIDatasetGenerator(
            cfg=roi_dataset_config,
        )

        val_roi_generator.output_dir = val_roi_dir
        val_roi_generator.roi_dir = os.path.join(val_roi_dir, "rois")
        val_roi_generator.metadata_path = os.path.join(val_roi_dir, "metadata.json")

        ensure_dir(val_roi_generator.roi_dir)

        val_metadata = val_roi_generator.generate_from_dataset(val_dataset)
        self.logger.info(f"Generated validation ROI dataset with {val_metadata['total_samples']} samples")

        train_roi_config = {**roi_dataset_config, "roi_dataset_dir": train_roi_dir}
        val_roi_config = {**roi_dataset_config, "roi_dataset_dir": val_roi_dir}

        train_roi_loader = ROIDatasetLoader(
            cfg=DictConfig(train_roi_config),
        )

        val_roi_loader = ROIDatasetLoader(
            cfg=DictConfig(val_roi_config),
        )
        if cnn_preprocessor:
            train_roi_loader.set_preprocessor(cnn_preprocessor)
            val_roi_loader.set_preprocessor(cnn_preprocessor)
        X_train, y_train = train_roi_loader.get_all_data()
        X_val, y_val = val_roi_loader.get_all_data()

        if len(X_train) == 0:
            self.logger.error("No training samples available for CNN training")
            return {}

        if len(X_val) == 0:
            self.logger.warning("No validation samples available, using a portion of training data for validation")
            # 如果没有验证数据，则从训练数据中分割出一部分作为验证数据
            val_split = self.cfg.get("dataset", {}).get("val_split", 0.2)
            indices = np.random.permutation(len(X_train))
            split_idx = int(len(indices) * (1 - val_split))
            train_idx, val_idx = indices[:split_idx], indices[split_idx:]

            X_val, y_val = X_train[val_idx], y_train[val_idx]
            X_train, y_train = X_train[train_idx], y_train[train_idx]

        self.logger.info(f"Training CNN with {len(X_train)} ROI samples, validating with {len(X_val)} samples")

        # 计算类别权重，处理不平衡数据
        class_weights = train_roi_loader.get_class_balance_weights()
        if class_weights:
            self.logger.info(f"Using class weights to handle imbalanced data: {class_weights}")

        # 训练CNN (注意数据已经预处理过)
        history = self.template_classifier.train(X_train, y_train, X_val, y_val, class_weights=class_weights)
        if history is None:
            history = {}

        # 保存训练完成的模型
        model_path = os.path.join(self.cfg.get("output_dir", "results"), "models")
        ensure_dir(model_path)
        self.template_classifier.save(os.path.join(model_path, "trained_model.h5"))

        # 保存训练配置
        roi_config_path = os.path.join(model_path, "roi_dataset_config.json")
        with open(roi_config_path, 'w') as f:
            roi_config_dict = dict(roi_dataset_config)
            json.dump(roi_config_dict, f, indent=2)

        self.logger.info("CNN training completed using ROI-based approach")

        return history

def main(cfg: DictConfig):
    logger = get_logger(__name__)
    try:
        pipeline = MetroTrainPipeline(cfg, logger)
        pipeline.run()
        logger.info("Training pipeline completed successfully")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

