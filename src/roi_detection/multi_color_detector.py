from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import cv2
from omegaconf import DictConfig
import logging
import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict

from src.data.dataset import MetroDataset
from src.preprocessing.roi_preprocessor import ROIParamOptimizerPreprocessor
from src.roi_detection.base_detector import BaseDetector
from utils.utils import ensure_dir, get_logger

class MultiColorDetector(BaseDetector):
    def __init__(self, cfg: DictConfig, logger: Optional[logging.Logger] = None):
        super().__init__(cfg, logger)

        self.min_area = cfg.get("min_area", 300)
        self.max_area = cfg.get("max_area", 20000)
        self.min_aspect_ratio = cfg.get("min_aspect_ratio", 0.5)
        self.max_aspect_ratio = cfg.get("max_aspect_ratio", 2.0)

        self.color_params = self._load_color_params(cfg)
        self.threshold_error_dict = cfg.get("threshold_error_dict", {})
        self.debug = cfg.get("debug", False)

        self.params_dir = cfg.get("params_dir", "")

    def _load_color_params(self, cfg: DictConfig) -> Dict:
        params_dir = cfg.get("params_dir", "")
        params_path = os.path.join(params_dir, "color_params.json")
        if params_path and os.path.exists(params_path):
            try:
                self.logger.info(f"Loading color parameters from {params_path}")
                with open(params_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load color parameters from {params_path}: {e}")

        self.logger.info("Using default color parameters for metro lines")
        return {}  
    def set_preprocessor(self, preprocessor: ROIParamOptimizerPreprocessor):
        super().set_preprocessor(preprocessor)
        if self.preprocessor is None:
            raise ValueError("Preprocessor is not set")
        if isinstance(self.preprocessor, ROIParamOptimizerPreprocessor):
            self.color_space = self.preprocessor.get_color_space()
        else:
            raise ValueError(f"Preprocessor {self.preprocessor} is not a ROIParamOptimizerPreprocessor")
            

    def detect(self, image: np.ndarray, has_visualize: bool = True) -> List[Dict[str, Any]]:
        if isinstance(self.preprocessor, ROIParamOptimizerPreprocessor):
            img_hsv = self.preprocessor.preprocess(image, has_visualize)
        else:
            raise ValueError(f"Preprocessor {self.preprocessor} is not a ROIParamOptimizerPreprocessor")
        detections = []

        for line_id in self.color_params:
            mask = self._extract_line_mask(img_hsv, line_id)
            boxes = self._extract_boxes_from_mask(mask)
            for x1, y1, x2, y2, conf in boxes:
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "line_id": line_id,
                    "confidence": conf
                })

        return self._nms_by_line(detections)


    def _extract_line_mask(self, img_hsv: np.ndarray, line_id: str) -> np.ndarray:
        params = self.color_params[line_id]
        lower = np.maximum(0, np.array(params["hsv_lower"]) - self.threshold_error_dict.get(line_id, 0))
        upper = np.minimum(255, np.array(params["hsv_upper"]) + self.threshold_error_dict.get(line_id, 0))
        lower = np.clip(lower, [0, 0, 0], [179, 255, 255])
        upper = np.clip(upper, [0, 0, 0], [179, 255, 255])
        # create initial color mask
        base_mask = cv2.inRange(img_hsv, lower, upper)
        H, W = base_mask.shape
        k = max(3, min(18, int(min(H, W) * 0.05)))
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        opened  = cv2.morphologyEx(base_mask, cv2.MORPH_OPEN,  ker)
        closed  = cv2.morphologyEx(opened,    cv2.MORPH_CLOSE, ker)
        dilated = cv2.dilate(closed, ker, iterations=1)

        def largest_cc_area(mask):
            cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts: return 0
            return max(cv2.contourArea(c) for c in cnts)

        orig_cnts,_ = cv2.findContours(base_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        over_eroded = False
        if orig_cnts:
            area_orig = largest_cc_area(base_mask)
            area_proc = largest_cc_area(dilated)
            num_orig  = len(orig_cnts)
            num_proc,_ = cv2.connectedComponents(dilated)
            if (area_orig > 0 and area_proc / area_orig < 0.2) or \
            (num_orig > 3 and num_proc < num_orig * 0.3):
                over_eroded = True

        if over_eroded:
            self.logger.info(f"[{line_id}] Over-eroded; fallback to conservative morph")
            small_ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_cons = cv2.morphologyEx(base_mask, cv2.MORPH_OPEN, small_ker, iterations=1)
            mask_cons = cv2.morphologyEx(mask_cons, cv2.MORPH_CLOSE, small_ker, iterations=1)
            final_mask = cv2.bitwise_or(mask_cons, dilated)
        else:
            final_mask = dilated

        return final_mask

    def _extract_boxes_from_mask(self, mask: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        
        height, width = mask.shape[:2]
        image_area = height * width
        
        dynamic_min_area = max(self.min_area, int(image_area * 0.01))  # at least 1% of the image
        dynamic_max_area = min(self.max_area, int(image_area * 0.1))   # at most 10% of the image

        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            if area < dynamic_min_area or area > dynamic_max_area:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            
            aspect_ratio = w / h if h > 0 else 0
            if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                continue
            
            fill_ratio = area / (w * h)
            if fill_ratio < 0.3:
                continue
            
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            circle_area = np.pi * r * r
            circularity = area / circle_area if circle_area > 0 else 0
            if circularity < 0.4:  
                continue
            
            perimeter = cv2.arcLength(cnt, True)
            complexity = perimeter * perimeter / (4 * np.pi * area) if area > 0 else float('inf')
            if complexity > 3.0:  
                continue
            
            confidence = (
                0.4 * fill_ratio +                
                0.3 * (1 - abs(0.75 - aspect_ratio)) +  
                0.3 * circularity                 
            )
            
            boxes.append((x, y, x + w, y + h, confidence))
        
        return boxes

    def _nms_by_line(self, detections: List[Dict[str, Any]], iou_thresh=0.5) -> List[Dict[str, Any]]:
        grouped = defaultdict(list)
        for det in detections:
            grouped[det['line_id']].append(det)

        final = []
        for line_id, group in grouped.items():
            final.extend(self._apply_nms_for_group(group, iou_thresh))
        return final

    def _apply_nms_for_group(self, regions: List[Dict[str, Any]], iou_threshold=0.5) -> List[Dict[str, Any]]:
        if not regions:
            return []

        boxes = np.array([r["bbox"] + [r["confidence"]] for r in regions])
        scores = boxes[:, 4]
        indices = np.argsort(scores)[::-1]
        keep = []

        while indices.size > 0:
            i = indices[0]
            keep.append(i)

            overlaps = self._calculate_iou(boxes[i, :4], boxes[indices[1:], :4])
            inds = np.where(overlaps <= iou_threshold)[0]
            indices = indices[inds + 1]

        return [regions[i] for i in keep]

    def _calculate_iou(self, box, boxes):
        area_box = (box[2] - box[0]) * (box[3] - box[1])
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        xx1 = np.maximum(box[0], boxes[:, 0])
        yy1 = np.maximum(box[1], boxes[:, 1])
        xx2 = np.minimum(box[2], boxes[:, 2])
        yy2 = np.minimum(box[3], boxes[:, 3])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        iou = inter / (area_box + area_boxes - inter + 1e-6)
        return iou

    def update_params(self, params: Dict[str, Any]) -> None:
        if 'min_area' in params:
            self.min_area = params['min_area']
        if 'max_area' in params:
            self.max_area = params['max_area']
        if 'min_aspect_ratio' in params:
            self.min_aspect_ratio = params['min_aspect_ratio']
        if 'max_aspect_ratio' in params:
            self.max_aspect_ratio = params['max_aspect_ratio']
        if 'color_params' in params:
            self.color_params.update(params['color_params'])

        self.logger.info("MultiColorDetector parameters updated")

    def optimize_color_parameters(self,dataset: MetroDataset, logger=None, visualize=False)->Dict[str, Any]:
        """
        Optimize color parameters based on training data.
        
        Args:
            dataset: Dataset with ground truth annotations, must implement:
                    - __len__() method
                    - get_image_with_annotations(idx) method that returns (image, annotations)
            logger: Optional logger
            visualize: Whether to visualize the dominant color extraction process
            
        Returns:
            Dictionary of optimized color parameters
        """
        import cv2
        import numpy as np
        
        logger = logger or get_logger(__name__)
        logger.info("Starting color parameter optimization...")
        
        optimized_params = {}
        
        color_samples = {}
        
        rois = []
        for idx in range(len(dataset)):
            image, annotations = dataset.get_image_with_annotations(idx)
            if image.dtype == np.float32 and image.max() <= 1.0:
                image_cv = (image * 255).astype(np.uint8)
            else:
                image_cv = image.astype(np.uint8)
            
            if len(image_cv.shape) == 3 and image_cv.shape[2] == 3:
                bgr = image_cv
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            else:
                continue
            for annotation in annotations:
                x1, y1, x2, y2 = annotation[:4]
                roi_hsv = hsv[y1:y2, x1:x2]
                roi_bgr = bgr[y1:y2, x1:x2]
                line_id = annotation[4]
                try:
                    dominant_hsv = self.extract_dominant_hsv(roi_hsv, K=3)
                    if line_id not in color_samples:
                        color_samples[line_id] = []
                    color_samples[line_id].append(dominant_hsv)
                    if visualize:
                        visualize_dominant_color(image, roi_bgr, dominant_hsv, line_id, x1, y1, x2, y2)
                except Exception as e:
                    logger.error(f"Error processing ROI ")
        
        for line_id, samples in color_samples.items():
            if not samples:
                continue
            
            samples_array = np.array(samples)
            
            h_values = samples_array[:, 0]
            if np.max(h_values) - np.min(h_values) > 90:
                h_values_adjusted = h_values.copy()
                if np.median(h_values) < 90:
                    h_values_adjusted[h_values > 90] -= 180
                else:
                    h_values_adjusted[h_values < 90] += 180
                
                avg_h = np.mean(h_values_adjusted)
                if avg_h < 0:
                    avg_h += 180
                elif avg_h > 180:
                    avg_h -= 180
            else:
                avg_h = np.mean(h_values)
            
            avg_s = np.mean(samples_array[:, 1])
            avg_v = np.mean(samples_array[:, 2])
            
            std_h = np.std(h_values)
            std_s = np.std(samples_array[:, 1])
            std_v = np.std(samples_array[:, 2])
            
            optimized_params[str(line_id)] = {
                "hsv_mean": (int(avg_h), int(avg_s), int(avg_v)),
                "hsv_std": (int(std_h), int(std_s), int(std_v)),
                "hsv_lower": [max(0, int(avg_h - 2 * std_h)), 
                            max(0, int(avg_s - 2 * std_s)), 
                            max(0, int(avg_v - 2 * std_v))],
                "hsv_upper": [min(180, int(avg_h + 2 * std_h)), 
                            min(255, int(avg_s + 2 * std_s)), 
                            min(255, int(avg_v + 2 * std_v))]
            }
        self.logger.info(f"Optimization finished, saving parameters to {self.params_dir}")
        self.save_params(optimized_params)
        return optimized_params 
    def extract_dominant_hsv(self, roi_hsv: np.ndarray, K: int = 3, attempts: int = 10) -> Tuple[int, int, int]:
        """
        From a single ROI's HSV pixels, extract the dominant HSV color.

        Args:
            roi_hsv: The pixels in the HSV space (h, w, 3)
            K: The number of clusters (clusters)
            attempts: The number of restarts for kmeans, selecting the best centroid

        Returns:
            hsv_dominant: The dominant HSV color (H, S, V)
        """

        pixels = roi_hsv.reshape(-1, 3).astype(np.float32) 

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100,    
            0.2     
        )

        ret, labels, centers = cv2.kmeans(
            pixels,
            K,
            None, # type: ignore
            criteria,
            attempts,
            flags=cv2.KMEANS_PP_CENTERS
        ) #type: ignore

        _, counts = np.unique(labels, return_counts=True)

        dominant_idx = np.argmax(counts)
        dominant_center = centers[dominant_idx] 

        return tuple(map(int, dominant_center)) #type: ignore
    def save_params(self, params: Dict[str, Any]) -> bool:
        """
        Save parameters to a JSON file. Creates the file if it doesn't exist,
        or overwrites it if it already exists.
        
        Args:
            params: Dictionary of parameters to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            ensure_dir(self.params_dir)
            params_path = os.path.join(self.params_dir, "color_params.json")
            
            # Check if file exists and log appropriate message
            if os.path.exists(params_path):
                self.logger.info(f"Overwriting existing parameters file: {params_path}")
            else:
                self.logger.info(f"Creating new parameters file: {params_path}")
                
            # Write parameters to file with pretty formatting
            with open(params_path, "w") as f:
                json.dump(params, f, indent=4, sort_keys=True)
                
            self.logger.info(f"Successfully saved parameters for {len(params)} metro lines")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save parameters: {e}")
            return False
def visualize_dominant_color(img, roi, hsv_color, line_id, x1, y1, x2, y2):

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,  # type: ignore
                                    edgecolor='g', facecolor='none', linewidth=2)) 
    plt.title(f"Original Image with Line {line_id} ROI")

    plt.subplot(1, 3, 2)
    plt.imshow(roi)
    plt.title(f"ROI of Line {line_id} BGR")
    
    plt.subplot(1, 3, 3)
    color_patch = np.ones((100, 100, 3), dtype=np.uint8)
    h, s, v = hsv_color
    color_patch_hsv = np.full((100, 100, 3), (h, s, v), dtype=np.uint8)
    color_patch = cv2.cvtColor(color_patch_hsv, cv2.COLOR_HSV2BGR)
    plt.imshow(color_patch)
    plt.title(f"Dominant Color\nHSV: ({h}, {s}, {v})")
    
    plt.tight_layout()
    plt.show()



def visualize_detection_steps(detector: MultiColorDetector, image: np.ndarray):

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    if isinstance(detector.preprocessor, ROIParamOptimizerPreprocessor):
        img_preprocessed = detector.preprocessor.preprocess(image, has_visualize=True)
    else:
        raise ValueError(f"Preprocessor {detector.preprocessor} is not a ROIParamOptimizerPreprocessor")
    
    axes[0].imshow(image)
    axes[0].set_title("1. Original Image BGR")
    axes[0].axis('off')


    img_bgr = image
    img_hsv_preprocessed = img_preprocessed
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    axes[1].imshow(img_hsv)
    axes[1].set_title("2. Original HSV")
    axes[1].axis('off')

    axes[2].imshow(img_hsv_preprocessed)
    axes[2].set_title("3. Preprocessed HSV")
    axes[2].axis('off')

    line_id = "12" if "12" in detector.color_params else list(detector.color_params.keys())[0]
    
    params = detector.color_params[line_id]
    lower = np.maximum(0, np.array(params["hsv_lower"]) - detector.threshold_error_dict.get(line_id, 0))
    upper = np.minimum(255, np.array(params["hsv_upper"]) + detector.threshold_error_dict.get(line_id, 0))
    

    original_mask = cv2.inRange(img_hsv_preprocessed, lower, upper)
    axes[3].imshow(original_mask, cmap='gray')
    axes[3].set_title(f"4. Original Mask (Line {line_id})")
    axes[3].axis('off')
    
    
    finial_mask = detector._extract_line_mask(img_hsv_preprocessed, line_id)
    
    axes[4].imshow(finial_mask, cmap='gray')
    axes[4].set_title(f"5. Final Mask")
    axes[4].axis('off')
    
    contour_img = image.copy()
    contours = detector._extract_boxes_from_mask(finial_mask)
    for i in range(len(contours)):
        x, y, xw, yh, confidence = contours[i]
        cv2.rectangle(contour_img, (x, y), (xw, yh), (0, 255, 0), 2)
    axes[5].imshow(contour_img)
    axes[5].set_title(f"6. All Contours ({len(contours)})")
    axes[5].axis('off')
    
    plt.show()
    
    print(f"检测线路 {line_id} 统计:")
    print(f"  - 原始掩码非零像素: {np.count_nonzero(original_mask)}")
    print(f"  - 清洗后掩码非零像素: {np.count_nonzero(finial_mask)}")
    print(f"  - 识别到的轮廓数: {len(contours)}")
    print(f"  - 筛选后的框数: {len(contours)}")
    print(f"  - 最终检测结果数: {len(contours)}")
    

def refine_line_mask(mask: np.ndarray) -> np.ndarray:
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (18, 18))
    close_kernel = open_kernel
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))

    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel)
    cleaned = cv2.dilate(closed, dilate_kernel, iterations=1)

    return cleaned