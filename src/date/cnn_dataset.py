import os
import json
import numpy as np
import cv2
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from utils.utils import get_logger, ensure_dir
from src.data.dataset import MetroDataset

class DigitFeatureExtractor:
    """
    专注于提取数字特征的预处理类，忽略颜色信息
    """
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.target_size = tuple(cfg.get("target_size", [32, 32]))
        self.binarize = cfg.get("binarize", True) 
        self.threshold_method = cfg.get("threshold_method", "adaptive")  
        self.morphology = cfg.get("morphology", False) 
        self.invert = cfg.get("invert", False)  
        
    def process(self, image: np.ndarray) -> np.ndarray:
        # 确保图像为灰度图
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 调整大小
        resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        
        # 二值化处理
        if self.binarize:
            if self.threshold_method == "adaptive":
                resized = cv2.adaptiveThreshold(
                    resized, 255, 
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 2
                )
            else: 
                _, resized = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 形态学操作（可选）
        if self.morphology:
            kernel = np.ones((2, 2), np.uint8)
            resized = cv2.morphologyEx(resized, cv2.MORPH_CLOSE, kernel)
        
        # 反转图像（可选）
        if self.invert:
            resized = cv2.bitwise_not(resized)
        
        # 归一化到0-1范围
        normalized = resized.astype(np.float32) / 255.0
        
        # 添加通道维度 (H, W) -> (H, W, 1)
        return normalized[..., np.newaxis]


class CNNTrainDatasetGenerator:
    """
    生成专注于数字特征的CNN训练数据集
    """
    
    def __init__(self, cfg: DictConfig, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger(__name__)
        self.cfg = cfg
        
        # 输出目录配置
        self.output_dir = self.cfg.get("cnn_dataset_dir", os.path.join(self.cfg.get("output_dir", "results"), "digit_cnn_dataset"))
        self.image_dir = os.path.join(self.output_dir, "images")
        self.metadata_path = os.path.join(self.output_dir, "metadata.json")
        self.train_val_split_path = os.path.join(self.output_dir, "train_val_split.json")
        
        # 创建输出目录
        ensure_dir(self.output_dir)
        ensure_dir(self.image_dir)
        
        # 预处理配置
        self.preprocessor = DigitFeatureExtractor(cfg.get("preprocessor", {}))
        
        # 训练集配置
        self.test_size = self.cfg.get("test_size", 0.2)
        self.val_size = self.cfg.get("val_size", 0.2)
        self.random_seed = self.cfg.get("random_seed", 42)
        
        self.logger.info(f"CNNTrainDatasetGenerator initialized with output_dir={self.output_dir}")
    
    def generate_from_dataset(self, dataset: MetroDataset) -> Dict[str, Any]:
        """
        从原始数据集生成数字特征数据集
        """
        self.logger.info(f"Generating digit CNN dataset from {len(dataset)} images")
        
        metadata = {
            "samples": [],
            "class_distribution": {},
            "total_samples": 0,
            "creation_params": {
                "test_size": self.test_size,
                "val_size": self.val_size,
                "random_seed": self.random_seed,
                "preprocessor": dict(self.cfg.get("preprocessor", {}))
            }
        }
        
        for idx in range(len(dataset)):
            image, annotations = dataset.get_image_with_annotations(idx)
            image_id = dataset.df.iloc[idx]['image_id']
            
            # 处理每个标注的数字区域
            for ann_idx, ann in enumerate(annotations):
                x1, y1, x2, y2, class_id = ann
                
                # 提取数字区域
                digit_roi = image[y1:y2, x1:x2]
                if digit_roi.size == 0 or digit_roi.shape[0] < 5 or digit_roi.shape[1] < 5:
                    self.logger.warning(f"Invalid digit ROI in image {image_id}: {ann}")
                    continue
                
                # 提取数字特征
                processed_digit = self.preprocessor.process(digit_roi)
                
                # 保存处理后的图像
                image_filename = f"digit_{image_id}_{ann_idx}.png"
                image_path = os.path.join(self.image_dir, image_filename)
                
              
                img_to_save = (processed_digit * 255).astype(np.uint8)
                if len(img_to_save.shape) == 3:
                    img_to_save = img_to_save.squeeze(-1) 
                cv2.imwrite(image_path, img_to_save)
                
                # 添加到元数据
                sample_info = {
                    "id": f"{str(image_id)}_{int(ann_idx)}",
                    "image_id": str(image_id),
                    "image_file": str(image_filename),
                    "class_id": int(class_id),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "is_background": False
                }
                
                metadata["samples"].append(sample_info)
                
                # 更新类别分布
                class_str = str(int(class_id))
                metadata["class_distribution"][class_str] = metadata["class_distribution"].get(class_str, 0) + 1
        
      
        metadata["total_samples"] = len(metadata["samples"])
        
        
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Generated digit CNN dataset with {metadata['total_samples']} samples")
        self.logger.info(f"Class distribution: {metadata['class_distribution']}")
        
        return metadata
    
    def generate_training_data(self, val_ratio: float = 0.2, test_ratio: float = 0.2) -> None:
        """
        生成训练集、验证集和测试集的分割
        """
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
            
        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)
        
        samples = metadata["samples"]
        
        # 按类别分层分割
        class_samples = {}
        for sample in samples:
            class_id = sample["class_id"]
            if class_id not in class_samples:
                class_samples[class_id] = []
            class_samples[class_id].append(sample)
        
        train_samples = []
        val_samples = []
        test_samples = []
        
        for class_id, samples in class_samples.items():
            # 先分割出测试集
            train_val, test = train_test_split(
                samples, 
                test_size=test_ratio, 
                random_state=self.random_seed,
                shuffle=True
            )
            
            # 再从剩余样本中分割出验证集
            val_ratio_adj = val_ratio / (1 - test_ratio)  # 调整验证集比例
            train, val = train_test_split(
                train_val, 
                test_size=val_ratio_adj, 
                random_state=self.random_seed,
                shuffle=True
            )
            
            train_samples.extend(train)
            val_samples.extend(val)
            test_samples.extend(test)
        
        # 保存分割结果
        split_data = {
            "train": train_samples,
            "val": val_samples,
            "test": test_samples,
            "split_ratios": {
                "val": val_ratio,
                "test": test_ratio
            },
            "random_seed": self.random_seed
        }
        
        with open(self.train_val_split_path, 'w') as f:
            json.dump(split_data, f, indent=2)
        
        self.logger.info(f"Training data split generated:")
        self.logger.info(f"  Training samples: {len(train_samples)}")
        self.logger.info(f"  Validation samples: {len(val_samples)}")
        self.logger.info(f"  Test samples: {len(test_samples)}")


class DigitCNNDataLoader:
    """
    加载数字CNN数据集并提供训练、验证和测试数据
    """
    
    def __init__(self, cfg: DictConfig, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger(__name__)
        self.cfg = cfg
        
        # 数据集目录配置
        self.dataset_dir = self.cfg.get("cnn_dataset_dir", os.path.join(self.cfg.get("output_dir", "results"), "digit_cnn_dataset"))
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.metadata_path = os.path.join(self.dataset_dir, "metadata.json")
        self.split_path = os.path.join(self.dataset_dir, "train_val_split.json")
        
        # 加载元数据
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
            
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        # 加载分割数据
        if not os.path.exists(self.split_path):
            raise FileNotFoundError(f"Split data file not found: {self.split_path}")
            
        with open(self.split_path, 'r') as f:
            self.split_data = json.load(f)
            
        self.logger.info(f"DigitCNNDataLoader initialized with dataset_dir={self.dataset_dir}")
    
    def _load_images(self, samples: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """加载图像和标签"""
        images = []
        labels = []
        
        for sample in samples:
            image_path = os.path.join(self.image_dir, sample["image_file"])
            if not os.path.exists(image_path):
                self.logger.warning(f"Image file not found: {image_path}")
                continue
                
            try:
                # 以灰度模式加载图像
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    self.logger.warning(f"Failed to read image: {image_path}")
                    continue
                
                # 添加通道维度
                image = image[..., np.newaxis]
                images.append(image)
                labels.append(sample["class_id"])
            except Exception as e:
                self.logger.error(f"Error loading image {image_path}: {e}")
        
        if not images:
            self.logger.warning("No valid images found")
            return np.array([]), np.array([])
            
        return np.array(images), np.array(labels)
    
    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取训练数据"""
        return self._load_images(self.split_data["train"])
    
    def get_val_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取验证数据"""
        return self._load_images(self.split_data["val"])
    
    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取测试数据"""
        return self._load_images(self.split_data["test"])
    
    def get_all_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取所有数据"""
        return self._load_images(self.metadata["samples"])