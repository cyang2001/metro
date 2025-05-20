import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
import scipy.io as sio
from typing import List, Optional, Tuple, Dict, Any, Union
from omegaconf import DictConfig
from tqdm import tqdm
from src.data.dataset import MetroDataset
from src.preprocessing.roi_preprocessor import ROIParamOptimizerPreprocessor
from src.preprocessing.template_preprocessor import TemplatePreprocessor
from src.roi_detection.multi_color_detector import MultiColorDetector, visualize_detection_steps
from src.classification.template_classifier import TemplateClassifier
from utils.utils import get_logger, ensure_dir, save_confusion_matrix, visualize_detection

class MetroTestPipeline:
    """
    Paris Metro Line Recognition Test Pipeline
    
    Processes test images to detect and classify metro line signs, and evaluates accuracy.
    """
    
    def __init__(self, cfg: DictConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize the test pipeline.
        
        Args:
            cfg: Configuration object
            logger: Optional logger instance
        """
        self.logger = logger or get_logger(__name__)
        self.cfg = cfg
        self._init_components()
    
    def _init_components(self):
        """
        Initialize pipeline components.
        """
        try:
            self.logger.info("Initializing test components...")
            
            self.roi_param_optimizer_preprocessor = ROIParamOptimizerPreprocessor(
                cfg=self.cfg.preprocessing.roi_param_optimizer
            )

            self.template_preprocessor = TemplatePreprocessor(
                cfg=self.cfg.preprocessing.template
            )
            
            self.multi_color_detector = MultiColorDetector(
                cfg=self.cfg.roi_detection
            )

            self.template_classifier = TemplateClassifier(
                cfg=self.cfg.classification
            )
            
            self.template_classifier.set_preprocessor(self.template_preprocessor)
            self.multi_color_detector.set_preprocessor(self.roi_param_optimizer_preprocessor)

            self.logger.info("Test components initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    
    def run(self):
        """
        Run the test pipeline.
        
        Process test images, detect ROIs, classify them, and evaluate results.
        """
        self.logger.info("=== Starting Test Pipeline ===")
        
        test_dataset = MetroDataset(
            cfg=self.cfg.dataset,
            mode='test',
        )
        
        self.logger.info(f"Test dataset loaded: {len(test_dataset)} samples")
        
        results = []
        predictions = []
        ground_truth = []
        has_visualize = False
        
        for idx in tqdm(range(len(test_dataset)), total=len(test_dataset), 
                        desc="Processing images", unit="img"):
            image, annotations, image_id = test_dataset.get_image_with_annotations(idx)
            self.logger.info(f"Processing image ID: {image_id}")
            #if image_id != 4:
            #    continue
            random_int = np.random.randint(0, 1)
            if not has_visualize and random_int == 0:
                detected_rois = self.multi_color_detector.detect(image, has_visualize=has_visualize)
                has_visualize = True
            else:
                detected_rois = self.multi_color_detector.detect(image, has_visualize=has_visualize)
            self.logger.info(f"Detected {len(detected_rois)} ROIs")
            gt_classes = [(ann[4], ann[:4]) for ann in annotations]

            #if 7 in [cls for cls, _ in gt_classes]:
            #    visualize_detection_steps(self.multi_color_detector, image)
            #visualize_detection_steps(self.multi_color_detector, image)
            detected_classes = []
            for roi in detected_rois:
                x1, y1, x2, y2 = roi['bbox']
                detected_classes.append((roi['line_id'], roi['bbox'], roi['confidence']))

                #roi_img = image[y1:y2, x1:x2]
                
                #if roi_img.size == 0 or roi_img.shape[0] == 0 or roi_img.shape[1] == 0:
                #    self.logger.warning(f"Invalid ROI: {roi}")
                #    continue
                
                #class_id, confidence = self.template_classifier.predict(roi_img)
                #if class_id == -1:
                #    continue
                #detected_classes.append((class_id, roi['bbox'], confidence))

            
            
            image_result = {
                'image_id': int(image_id),
                'detected': [(cls, bbox, conf) for cls, bbox, conf in detected_classes],
                'ground_truth': gt_classes
            }
            results.append(image_result)
            
            for cls, _, _ in detected_classes:
                predictions.append(cls)
            for cls, _ in gt_classes:
                ground_truth.append(cls)
            
            if self.cfg.mode.get("view_images", False):
                self._visualize_results(image, detected_classes, gt_classes, image_id)
        
        #self._calculate_metrics(results, predictions, ground_truth)
        
        self._save_results(results)
        
        self.logger.info("=== Test Completed ===")
    
    def _visualize_results(self, image: np.ndarray, detections: List[Tuple], 
                          ground_truth: List[Tuple], image_id: int):
        """
        Visualize detection and classification results.
        
        Args:
            image: Original image
            detections: List of (class_id, bbox, confidence)
            ground_truth: List of (class_id, bbox)
            image_id: Image identifier
        """
        det_boxes = [(box[0], box[1], box[2], box[3], cls) for cls, box, _ in detections]
        
        gt_boxes = [(box[0], box[1], box[2], box[3], cls) for cls, box in ground_truth]
        
        if self.cfg.mode.get("save_visualization", False):
            output_dir = os.path.join(self.cfg.mode.get("output_dir", "results"), "test_visualize")
            ensure_dir(output_dir)
            save_path = os.path.join(output_dir, f"test_result_{image_id}.png")
        else:
            save_path = None
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Detections")
        
        for x1, y1, x2, y2, cls in det_boxes:
            rect = Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(rect)
            plt.text(x1, y1-5, f"Line {cls}", color='red', fontsize=8, 
                     bbox=dict(facecolor='white', alpha=0.7))
        
        plt.subplot(1, 2, 2)
        plt.imshow(image)
        plt.title("Ground Truth")
        
        for x1, y1, x2, y2, cls in gt_boxes:
            rect = Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='green', linewidth=2)
            plt.gca().add_patch(rect)
            plt.text(x1, y1-5, f"Line {cls}", color='green', fontsize=8,
                     bbox=dict(facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            self.logger.info(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def _calculate_metrics(self, results: List[Dict], predictions: List[int], 
                          ground_truth: List[int]):
        """
        Calculate evaluation metrics.
        
        Args:
            results: List of detection results per image
            predictions: List of predicted classes
            ground_truth: List of ground truth classes
        """
        self.logger.info("Calculating evaluation metrics...")
        
        from sklearn.metrics import confusion_matrix, classification_report
        
        try:
            if predictions and ground_truth:
                all_classes = sorted(list(set(predictions + ground_truth)))
                
                cm = confusion_matrix(ground_truth, predictions, labels=all_classes)
                report = classification_report(ground_truth, predictions, 
                                             labels=all_classes, output_dict=True)
                
                self.logger.info(f"Classification report:\n{classification_report(ground_truth, predictions, labels=all_classes)}")
                
                output_dir = os.path.join(self.cfg.get("output_dir", "results"), "plots")
                ensure_dir(output_dir)
                
                save_confusion_matrix(
                    cm,
                    all_classes,
                    os.path.join(output_dir, "test_confusion_matrix.png"),
                    title="Test Confusion Matrix",
                    normalize=True
                )
                
                correct = sum(1 for p, gt in zip(predictions, ground_truth) if p == gt)
                total = len(ground_truth)
                accuracy = correct / total if total > 0 else 0
                
                self.logger.info(f"Test accuracy: {accuracy:.4f} ({correct}/{total})")
                
                metrics = {
                    'confusion_matrix': cm,
                    'classification_report': report,
                    'accuracy': accuracy
                }
                
                # Save metrics
                self._save_metrics(metrics)
                
            else:
                self.logger.warning("No predictions or ground truth available for evaluation")
                
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
    
    def _save_metrics(self, metrics: Dict):
        """
        Save evaluation metrics.
        
        Args:
            metrics: Dictionary of metrics
        """
        try:
            import json
            output_dir = os.path.join(self.cfg.get("output_dir", "results"), "metrics")
            ensure_dir(output_dir)
            
            # Save report as JSON
            with open(os.path.join(output_dir, "test_metrics.json"), 'w') as f:
                # Convert numpy arrays to lists
                for k, v in metrics.items():
                    if hasattr(v, 'tolist'):
                        metrics[k] = v.tolist()
                json.dump(metrics, f, indent=4)
                
            self.logger.info(f"Metrics saved to {output_dir}")
                
        except Exception as e:
            self.logger.error(f"Error saving metrics: {e}")
    
    def _save_results(self, results: List[Dict]):
        """
        Save detection results.
        
        Args:
            results: List of detection results per image
        """
        try:
            output_dir = os.path.join(self.cfg.get("output_dir", "results"), "detections")
            ensure_dir(output_dir)
            
            # Convert to format suitable for MATLAB
            bd_data = []
            for result in results:
                image_id = result['image_id']

                if self.cfg.mode.get("save_detections", True):
                    for cls, bbox, _ in result['detected']:
                        x1, y1, x2, y2 = map(float, bbox)
                        bd_data.append([image_id, y1, y2, x1, x2, cls])
            
            # Save as MATLAB file
            mat_path = os.path.join(output_dir, "test_results.mat")
            sio.savemat(mat_path, {'BD': np.array(bd_data, dtype=np.float64)})
            
            self.logger.info(f"Results saved to {mat_path}")
            
            # Save as JSON (more readable)
            json_path = os.path.join(output_dir, "test_results.json")
            

            with open(json_path, 'w') as f:
                import json
                json.dump(results, f, indent=4, default=lambda x: str(x) if isinstance(x, np.ndarray) else x)
                
            self.logger.info(f"Results also saved to {json_path}")
                
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")

def main(cfg: DictConfig):
    """
    Main entry point for test pipeline.
    
    Args:
        cfg: Configuration object
    """
    logger = get_logger(__name__)
    logger.info("Initializing test pipeline")
    
    pipeline = MetroTestPipeline(cfg)
    pipeline.run()

def debug_roi_matching(image: np.ndarray, roi: Tuple[int, int, int, int], match_class: int, confidence: float):
    """
    Visualize an ROI in the full image and the cropped region.

    Args:
        image: full RGB image
        roi: (x1, y1, x2, y2)
        match_class: class ID returned by classifier
        confidence: matching confidence score
    """
    x1, y1, x2, y2 = roi
    roi_img = image[y1:y2, x1:x2]  # 注意：cv2 是 (y1:y2, x1:x2)

    plt.figure(figsize=(12, 5))

    # Full image with bounding box
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Full Image")
    plt.gca().add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', linewidth=2, fill=False))
    plt.text(x1, y1 - 10, f"Predicted: {match_class}, Conf: {confidence:.2f}", color='red')

    # ROI image
    plt.subplot(1, 2, 2)
    plt.imshow(roi_img)
    plt.title("Matched ROI")

    plt.tight_layout()
    plt.show()