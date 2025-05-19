"""
Author: @Chen YANG
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Union, Optional
from omegaconf import DictConfig
import glob
from pathlib import Path
from matplotlib.patches import Rectangle
import cv2

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(level)
        
        handler = logging.StreamHandler()
        handler.setLevel(level)
        
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        
        logger.propagate = False
    
    return logger

def ensure_dir(path: Union[str, Path]) -> None:
    try:
        path = Path(path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Error creating directory {path}: {e}")

def find_file(filename: str, file_dir: str) -> Optional[str]:
    logger = get_logger(__name__)
    
    if os.path.exists(os.path.join(file_dir, filename)):
        return os.path.join(file_dir, filename)
    else:
        return None

def draw_rectangle(img: np.ndarray, bbox: Tuple[int, int, int, int], label: Optional[str] = None, 
                  color: Tuple[float, float, float] = (1.0, 0, 0), thickness: int = 2) -> np.ndarray:
    if img.dtype != np.uint8:
        if img.dtype in [np.float32, np.float64] and np.max(img) <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    
    img_copy = img.copy()
    
    color_bgr = [int(c * 255) for c in color[::-1]]  
    
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), color_bgr, thickness)
    
    if label:
        cv2.putText(img_copy, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, color_bgr, 1, cv2.LINE_AA)
    
    return img_copy

def plot_confusion_matrix(cm: np.ndarray, classes: List[str], 
                         normalize: bool = False, 
                         title: str = 'Confusion matrix',
                         save_path: Optional[str] = None) -> None:
    plt.figure(figsize=(10, 8))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path)
    
    plt.show()

def save_results_to_mat(results: List[List[Any]], output_file: str = 'myResults.mat') -> str:

    import scipy.io as sio
    import numpy as np
    results_array = np.array(results, dtype=np.float64)
    
    sio.savemat(output_file, {'BD': results_array})
    
    return output_file

def save_confusion_matrix(
    cm: np.ndarray,
    classes: List[int],
    output_path: str,
    title: str = 'Confusion Matrix',
    normalize: bool = False
) -> None:
    if normalize:
        row_sums = cm.sum(axis=1)
        row_sums = np.where(row_sums == 0, 1, row_sums)  
        cm = cm.astype('float') / row_sums[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues) # type: ignore
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    classes_str = [str(c) for c in classes]
    plt.xticks(tick_marks, classes_str, rotation=45)
    plt.yticks(tick_marks, classes_str)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    ensure_dir(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=100)
    plt.close()

def visualize_detection(
    image: np.ndarray,
    boxes: List[Tuple[int, int, int, int, int]],
    title: str = "Detection Results",
    show: bool = True,
    save_path: Optional[str] = None
) -> None:
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    
    vis_image = image.copy()
    
    plt.figure(figsize=(12, 8))
    
    plt.imshow(vis_image)
    
    cmap = plt.cm.get_cmap('tab10', 10)
    
    for box in boxes:
        x1, y1, x2, y2, class_id = box
        
        color_idx = class_id % 10
        color = cmap(color_idx)
        
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                        linewidth=2, edgecolor=color, facecolor='none')
        plt.gca().add_patch(rect)
        
        plt.text(x1, y1 - 5, f"Class {class_id}",
                color=color, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=100)
    
    if show:
        plt.show()
    else:
        plt.close()

def visualize_results(
    images: List[np.ndarray],
    true_labels: List[int],
    pred_labels: List[int],
    confidences: List[float],
    max_images: int = 20,
    title: str = "Classification Results",
    save_path: Optional[str] = None
) -> None:
    n_images = min(len(images), max_images)
    if n_images <= 0:
        return
    
    n_cols = min(5, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    plt.figure(figsize=(n_cols * 3, n_rows * 3))
    plt.suptitle(title, fontsize=16)
    
    for i in range(n_images):
        plt.subplot(n_rows, n_cols, i + 1)
        
        img = images[i]
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)
        
        plt.imshow(img)
        
        color = 'green' if pred_labels[i] == true_labels[i] else 'red'
        
        plt.title(f"True: {true_labels[i]}, Pred: {pred_labels[i]}\nConf: {confidences[i]:.2f}", 
                 color=color, fontsize=10)
        plt.axis('off')
    
    plt.tight_layout(rect=(0, 0, 1, 0.95))  
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=100)
    
    plt.show()

def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training History",
    save_path: Optional[str] = None
) -> None:
    plt.figure(figsize=(12, 5))
    plt.suptitle(title, fontsize=16)
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    if 'accuracy' in history:
        plt.plot(history['accuracy'], label='Train')
    if 'val_accuracy' in history:
        plt.plot(history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    if 'loss' in history:
        plt.plot(history['loss'], label='Train')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout(rect=(0, 0, 1, 0.95))      
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=100)
    
    plt.show()

def resize_with_aspect_ratio(
    image: np.ndarray,
    target_size: Tuple[int, int],
    pad_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    scale = min(target_h / h, target_w / w)
    
    new_h, new_w = int(h * scale), int(w * scale)
    
    resized = cv2.resize(image, (new_w, new_h))
    
    if len(image.shape) == 3:
        result = np.ones((target_h, target_w, 3), dtype=image.dtype) * np.array(pad_color, dtype=image.dtype)
        if image.dtype == np.float32 or image.dtype == np.float64:
            result = result / 255.0
    else:
        result = np.ones((target_h, target_w), dtype=image.dtype) * pad_color[0]
        if image.dtype == np.float32 or image.dtype == np.float64:
            result = result / 255.0
    
    pad_h = (target_h - new_h) // 2
    pad_w = (target_w - new_w) // 2
    
    if len(image.shape) == 3:
        result[pad_h:pad_h + new_h, pad_w:pad_w + new_w, :] = resized
    else:
        result[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    
    return result 