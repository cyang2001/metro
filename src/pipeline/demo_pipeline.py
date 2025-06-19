import logging
import os
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from omegaconf import DictConfig
from typing import List, Optional, Tuple, Dict, Any, Union
import cv2
from PIL import Image
import time
import re, scipy.io as sio, json

from src.roi_detection.multi_color_detector import MultiColorDetector
from src.data.dataset import MetroDataset
from src.preprocessing.roi_preprocessor import ROIParamOptimizerPreprocessor
from utils.utils import get_logger, ensure_dir
from src.preprocessing.CNN_preprocessor import CNNPreprocessor
from src.classification.CNN_classifier import CNNClassifier

class MetroDemoPipeline:
    """
    Paris Metro Line Recognition Demo Pipeline
    
    Processes single images or batches of images and visualizes the results.
    """
    
    def __init__(self, cfg: DictConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize the demo pipeline.
        
        Args:
            cfg: Configuration object
            logger: Optional logger instance
        """
        self.logger = logger or get_logger(__name__)
        self.cfg = cfg
        self._init_components()
        
        self.output_dir = self.cfg.get("output_dir", "results/demo")
        ensure_dir(self.output_dir)
        
        self.view_images = self.cfg.mode.demo.get("view_images", True)
        self.save_results = self.cfg.mode.demo.get("save_results", False)
        self.show_debug_info = self.cfg.mode.demo.get("show_debug_info", False)
        
        if self.save_results:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            self.output_dir = os.path.join(self.output_dir, timestamp)
            ensure_dir(self.output_dir)
            self.logger.info(f"Results will be saved to {self.output_dir}")
    
    def _init_components(self):
        """
        Initialize pipeline components.
        """
        try:
            self.logger.info("Initializing demo components...")
            
            self.roi_preprocessor = ROIParamOptimizerPreprocessor(
                cfg=self.cfg.preprocessing.roi_param_optimizer
            )
            self.roi_detector = MultiColorDetector(
                cfg=self.cfg.roi_detection
            )

            self.roi_detector.set_preprocessor(self.roi_preprocessor)

            # CNN classifier components
            self.cnn_preprocessor = CNNPreprocessor(
                cfg=self.cfg.preprocessing.cnn
            )
            self.cnn_classifier = CNNClassifier(
                cfg=self.cfg.classification.cnn
            )
            self.cnn_classifier.set_preprocessor(self.cnn_preprocessor)

            self.logger.info("Demo components initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def run(self):
        """
        Run the demo pipeline.
        
        Process single image or all images in a directory based on configuration.
        """
        self.logger.info("=== Starting Demo Pipeline ===")
        
        input_path = self.cfg.mode.demo.input_path
        batch_mode = self.cfg.mode.demo.get("batch_mode", False)
        
        if batch_mode:
            self._process_batch(input_path)
        else:
            self._process_single(input_path)
            
        self.logger.info("=== Demo Completed ===")
    
    def _process_single(self, image_path: str):
        """
        Process a single image.
        
        Args:
            image_path: Path to the image file
        """
        self.logger.info(f"Processing single image: {image_path}")
        
        if not os.path.exists(image_path):
            self.logger.error(f"Image not found: {image_path}")
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            start_time = time.time()
            image = self._load_image(image_path)
            roi_results = self.roi_detector.detect(image)
            detection_time = time.time() - start_time
            #self.logger.info(f"Detected {len(roi_results)} potential metro signs")
            
            # Classification stage
            processing_times = {
                'detection': detection_time,
            }

            cls_start = time.time()
            class_results = self._classify_rois(image, roi_results)
            # Remove duplicates
            class_results = self._filter_duplicate_rois(class_results)
            processing_times['classification'] = time.time() - cls_start

            results = class_results

            if self.view_images:
                self._visualize_results(image, results, os.path.basename(image_path), processing_times)
            
            if self.save_results:
                output_path = os.path.join(self.output_dir, f"demo_{os.path.basename(image_path)}")
                self._save_visualization(image, results, output_path, processing_times)
                self._save_detection_data(results, os.path.splitext(output_path)[0] + "_data.json")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """
        Load image from file.
        
        Args:
            image_path: Image file path
            
        Returns:
            original_image float32 [0,1]
        """

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image {image_path}")

        self.logger.info(f"Loaded image with shape {image.shape}")
        
        image = image.astype(np.float32) / 255.0

        return image
    
    def _classify_rois(self, image: np.ndarray, roi_results: List[Dict]) -> List[Dict]:
        """
        Classify detected ROIs.
        
        Args:
            image: Original image
            roi_results: ROI detection results
            
        Returns:
            Classification results list
        """
        results = []
        for roi in roi_results:
            try:
                # 提取ROI坐标
                bbox = roi["bbox"]
                x1, y1, x2, y2 = bbox
                
                # 确保坐标在图像范围内
                h, w = image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 <= x1 or y2 <= y1:
                    self.logger.warning(f"Invalid ROI coordinates: ({x1}, {y1}, {x2}, {y2})")
                    continue
                
                # 提取ROI区域
                roi_img = image[y1:y2, x1:x2]
                
                if roi_img.size == 0:
                    self.logger.warning(f"Empty ROI: ({x1}, {y1}, {x2}, {y2})")
                    continue
                
                # CNN classifier prediction
                class_id, confidence = self.cnn_classifier.predict(roi_img)
                
                if class_id != -1:
                    result = {
                        'bbox': (x1, y1, x2, y2),
                        'class_id': class_id,
                        'confidence': confidence,
                        'roi_confidence': roi.get('confidence', 0.0),
                        'line_id': class_id
                    }
                    results.append(result)
                    #self.logger.info(f"Detected metro line {class_id} with confidence {confidence:.4f}")
            except Exception as e:
                self.logger.error(f"Error classifying ROI: {e}")
                
        return results
    
    def _process_batch(self, directory: str):
        """
        Process all images in a directory.
        
        Args:
            directory: Directory containing images
        """
        self.logger.info(f"Processing images in directory: {directory}")
        
        if not os.path.isdir(directory):
            self.logger.error(f"Directory not found: {directory}")
            return
        
        # Get all image files
        image_files = []
        for ext in ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']:
            image_files.extend(glob.glob(os.path.join(directory, f"*.{ext}")))
        
        self.logger.info(f"Found {len(image_files)} images")
        
        if not image_files:
            self.logger.warning(f"No images found in {directory}")
            return
        
        # Process each image
        all_results = []
        for image_path in image_files:
            try:
                self.logger.info(f"Processing {os.path.basename(image_path)}")
                
                image_results = self._process_single(image_path)
                
                for result in image_results:
                    result['image_path'] = image_path
                    all_results.append(result)
                    
            except Exception as e:
                self.logger.error(f"Error processing image {image_path}: {e}")
        
        self._generate_summary(all_results)
        self._save_results(all_results)
    
    def _visualize_results(self, image: np.ndarray, results: List[Dict], title: str = "", processing_times: Optional[Dict] = None):
        """
        Visualize detection results.
        
        Args:
            image: Original image
            results: List of detection results
            title: Optional title for visualization
            processing_times: Optional processing time information
        """
        if not self.view_images:
            return
            
        plt.figure(figsize=(12, 8))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        
        # Color map for different classes
        cmap = plt.cm.get_cmap('tab10', 14)  
        
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            line_id = result['line_id']
            confidence = result['confidence']
            
            # Convert line_id to integer for colormap
            try:
                color_idx = int(line_id)
            except (ValueError, TypeError):
                color_idx = 0
                
            color = cmap(color_idx)
            
            # Draw rectangle
            rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2,
                            edgecolor=color, facecolor='none')
            plt.gca().add_patch(rect)
            
            label_text = f"Line {line_id} ({confidence:.2f})"
            plt.text(x1, y1-10, label_text,
                    color='white', fontsize=10, 
                    bbox=dict(facecolor=color, alpha=0.7))
        
        if processing_times is not None and self.show_debug_info:
            info_text = f"Detection: {processing_times['detection']:.3f}s"
            if 'classification' in processing_times:
                info_text += f", Class: {processing_times['classification']:.3f}s"
            plt.figtext(0.02, 0.02, info_text, color='black', 
                       backgroundcolor='white', fontsize=9)
        
        plt.title(f"Detection Results - {title}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def _save_visualization(self, image: np.ndarray, results: List[Dict], output_path: str, processing_times: Optional[Dict] = None):
        """
        Save visualization to file.
        
        Args:
            image: Original image
            results: List of detection results
            output_path: Path to save visualization
            processing_times: Optional processing time information
        """
        # Create figure
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        
        # Color map for different classes
        cmap = plt.cm.get_cmap('tab10', 14)
        

        # results结构最后要改一下
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            class_id = result['class_id']
            confidence = result['confidence']
            roi_confidence = result['confidence']
            #roi_confidence = result.get('roi_confidence', 0.0)
            #classification_confidence = result.get('classification_confidence', 0.0)

            # Convert line_id to integer for colormap
            try:
                color_idx = int(class_id)
            except (ValueError, TypeError):
                # If conversion fails, use a default value
                color_idx = 0
                
            color = cmap(color_idx)
            
            # Draw rectangle
            rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2,
                            edgecolor=color, facecolor='none')
            plt.gca().add_patch(rect)
            
            label_text = f"Line {class_id} ({confidence:.2f})"
            if self.show_debug_info:
                label_text += f"\nROI conf: {roi_confidence:.2f}"
                
            plt.text(x1, y1-10, label_text,
                    color='white', fontsize=10, 
                    bbox=dict(facecolor=color, alpha=0.7))
        

        
        plt.title(f"Detection Results")
        plt.axis('off')
        plt.tight_layout()
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        ensure_dir(output_dir)
        
        # Save figure
        plt.savefig(output_path, dpi=200)
        plt.close()
        
        self.logger.info(f"Visualization saved to {output_path}")

    def _save_results(self, results: List[Dict]):
        """
        Save detection results.
        
        Args:
            results: List of detection results per image
        """
        try:
            # Determine output directory (fallback to current working dir)
            out_dir = self.cfg.mode.demo.get("output_path") or os.getcwd()
            ensure_dir(out_dir)

            # Build BD rows: [n, x1, x2, y1, y2, class]
            pattern = re.compile(r"\((\d+)\)")
            bd_rows: List[List[float]] = []
            for r in results:
                img_path = r.get("image_path", "")
                m = pattern.search(os.path.basename(img_path))
                if m is None:
                    self.logger.warning(f"Skip '{img_path}': filename does not match '(n)' pattern")
                    continue

                n_val = int(m.group(1))
                x1, y1, x2, y2 = map(float, r["bbox"])
                class_id = int(r["class_id"])
                bd_rows.append([n_val, y1, y2, x1, x2, class_id])

            if not bd_rows:
                self.logger.error("_save_results: No BD rows produced, mat file not written")
                return

            bd_array = np.asarray(bd_rows, dtype=np.float64)

            mat_path = os.path.join(out_dir, "teams25.mat")
            sio.savemat(mat_path, {"BD": bd_array})
            self.logger.info(f"Saved {bd_array.shape[0]} detections to {mat_path}")

            # Optional JSON for inspection
            json_path = os.path.join(out_dir, "teams25.json")
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2, default=lambda o: o if isinstance(o, (int, float, str)) else str(o))
            self.logger.info(f"Raw results saved to {json_path}")
                
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")

    def _save_detection_data(self, results: List[Dict], output_path: str):
        """
        Save detection data to JSON file.
        
        Args:
            results: List of detection results
            output_path: Path to save data
        """
        import json
        
        serializable_results = []
        for r in results:
            serializable_result = {
                'bbox': list(r['bbox']),
                'class_id': int(r['class_id']),
                'confidence': float(r['confidence']),
                'roi_confidence': float(r.get('roi_confidence', 0.0)),
                'line_id': r.get('line_id', '')
            }
            serializable_results.append(serializable_result)
        
        # 保存到文件
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        self.logger.info(f"Detection data saved to {output_path}")
    
    def _generate_summary(self, all_results: List[Dict]):
        """
        Generate summary of batch processing results.
        
        Args:
            all_results: List of all detection results
        """
        if not all_results:
            self.logger.warning("No results to summarize")
            return
            
        self.logger.info("=== Summary ===")
        
        processed_images = set([r['image_path'] for r in all_results if 'image_path' in r])
        self.logger.info(f"Processed {len(processed_images)} images")
        self.logger.info(f"Detected {len(all_results)} metro signs")
        
        class_counts = {}
        for result in all_results:
            class_id = result['class_id']
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        self.logger.info("Detections by class:")
        for class_id in range(1, 15):  
            count = class_counts.get(class_id, 0)
            self.logger.info(f"  Line {class_id}: {count}")
        

        class_confidences = {}
        for result in all_results:
            class_id = result['class_id']
            if class_id not in class_confidences:
                class_confidences[class_id] = []
            class_confidences[class_id].append(result['confidence'])
        
        self.logger.info("Average confidence by class:")
        for class_id, confidences in class_confidences.items():
            avg_confidence = sum(confidences) / len(confidences)
            self.logger.info(f"  Line {class_id}: {avg_confidence:.4f}")
        
        if self.save_results:
            plt.figure(figsize=(12, 6))
            
            line_ids = list(range(1, 15)) 
            counts = [class_counts.get(line_id, 0) for line_id in line_ids]
            
            bars = plt.bar(
                [f"Line {line_id}" for line_id in line_ids],
                counts,
                color=[plt.cm.get_cmap('tab10', 14)((i-1) % 14) for i in line_ids]
            )
            
            for bar, count in zip(bars, counts):
                plt.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.1,
                    str(count),
                    ha='center',
                    va='bottom'
                )
            
            plt.title("Metro Line Detections Distribution")
            plt.xlabel("Line")
            plt.ylabel("Count")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            summary_path = os.path.join(self.output_dir, "class_distribution.png")
            plt.savefig(summary_path, dpi=200)
            plt.close()
            
            self.logger.info(f"Summary chart saved to {summary_path}")
            
            plt.figure(figsize=(12, 6))
            
            valid_line_ids = sorted(class_confidences.keys())
            avg_confidences = [np.mean(class_confidences[i]) for i in valid_line_ids]
            min_confidences = [min(class_confidences[i]) for i in valid_line_ids]
            max_confidences = [max(class_confidences[i]) for i in valid_line_ids]
            
            x = np.arange(len(valid_line_ids))
            plt.bar(
                x,
                avg_confidences,
                color=[plt.cm.get_cmap('tab10', 14)((i-1) % 14) for i in valid_line_ids],
                alpha=0.7
            )
            
            plt.errorbar(
                x,
                avg_confidences,
                yerr=[
                    [a - b for a, b in zip(avg_confidences, min_confidences)],
                    [b - a for a, b in zip(avg_confidences, max_confidences)]
                ],
                fmt='none',
                capsize=5,
                color='black'
            )
            
            plt.title("Classification Confidence by Line")
            plt.xlabel("Line")
            plt.ylabel("Confidence")
            plt.xticks(x, [f"Line {i}" for i in valid_line_ids])
            plt.ylim(0, 1.1)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            confidence_path = os.path.join(self.output_dir, "confidence_distribution.png")
            plt.savefig(confidence_path, dpi=200)
            plt.close()
            
            self.logger.info(f"Confidence chart saved to {confidence_path}")
            
            summary_data = {
                "processed_images": len(processed_images),
                "total_detections": len(all_results),
                "class_counts": {str(k): v for k, v in class_counts.items()},
                "avg_confidences": {str(k): float(np.mean(v)) for k, v in class_confidences.items()},
                "min_confidences": {str(k): float(min(v)) for k, v in class_confidences.items()},
                "max_confidences": {str(k): float(max(v)) for k, v in class_confidences.items()}
            }
            
            import json
            summary_json_path = os.path.join(self.output_dir, "summary.json")
            with open(summary_json_path, 'w') as f:
                json.dump(summary_data, f, indent=2)
                
            self.logger.info(f"Summary data saved to {summary_json_path}")

    # -----------------------------------------------------------------
    def _filter_duplicate_rois(self, detections: List[Dict], iou_thresh: float = 0.5) -> List[Dict]:
        """Apply simple NMS based on classifier confidence to keep one ROI per overlapping area."""
        if not detections:
            return detections

        # Sort detections by confidence (desc)
        dets = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        picked: List[Dict] = []

        while dets:
            best = dets.pop(0)
            picked.append(best)
            dets = [d for d in dets if self._iou(d['bbox'], best['bbox']) < iou_thresh]

        return picked

    def _iou(self, box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
        """Compute Intersection over Union of two bounding boxes."""
        x1_a, y1_a, x2_a, y2_a = box_a
        x1_b, y1_b, x2_b, y2_b = box_b

        inter_x1 = max(x1_a, x1_b)
        inter_y1 = max(y1_a, y1_b)
        inter_x2 = min(x2_a, x2_b)
        inter_y2 = min(y2_a, y2_b)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        area_a = (x2_a - x1_a) * (y2_a - y1_a)
        area_b = (x2_b - x1_b) * (y2_b - y1_b)
        union = area_a + area_b - inter_area
        if union == 0:
            return 0.0
        return inter_area / union

def main(cfg: DictConfig):
    """
    Main entry point for demo pipeline.
    
    Args:
        cfg: Configuration object
    """
    logger = get_logger(__name__)
    
    try:
        pipeline = MetroDemoPipeline(
            cfg=cfg,
            logger=logger
        )
        
        pipeline.run()
        
        logger.info("Demo pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Demo pipeline execution failed: {e}")
        import traceback
        logger.error(traceback.format_exc()) 