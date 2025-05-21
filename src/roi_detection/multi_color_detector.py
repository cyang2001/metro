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
        self.fill_ratio = cfg.get("fill_ratio", 0.75)
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
        """
        Detect regions of interest in an image based on color parameters.
        
        Args:
            image: Input image in BGR uint8 format. If image is float32 with values in [0,1],
                  it will be converted to uint8 internally.
            has_visualize: Whether to enable visualization during preprocessing.
                          Set to False during batch processing for performance.
        
        Returns:
            List[Dict[str, Any]]: List of detection results, each containing:
                - 'bbox': List[int] - Bounding box coordinates [x1, y1, x2, y2]
                - 'line_id': str - Detected line identifier
                - 'confidence': float - Detection confidence score
        
        Raises:
            ValueError: If preprocessor is not set or not of correct type.
        """
        if isinstance(self.preprocessor, ROIParamOptimizerPreprocessor):
            img_hsv = self.preprocessor.preprocess(image, has_visualize)
            #img_hsv = cv2.medianBlur(img_hsv, ksize=5)
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


    def _extract_line_mask(self, img_color_space: np.ndarray, line_id: str) -> np.ndarray:
        """
        Extract mask for a specific line ID.
        
        Args:
            img_color_space: Image in HSV or LAB color space
            line_id: Line identifier (e.g., '1', '4', etc.)
        
        Returns:
            np.ndarray: Binary mask with white pixels representing the detected color
        """
        height, width = img_color_space.shape[:2]

        base_kernel_size = max(3, int(min(height, width) * 0.005))  
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (base_kernel_size, base_kernel_size))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (base_kernel_size*2, base_kernel_size*2))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (base_kernel_size*3, base_kernel_size*3))

        params = self.color_params[line_id]
        
        threshold_error = self.threshold_error_dict.get(line_id, 0)



        # 根据色彩空间选择不同的处理流程
        # 暂时不考虑lab空间
        # HSV处理流程
        hsv_bonus = [0, 0, 0]
        
        hsv_mean = params["hsv_mean"]
        hsv_std = params["hsv_std"]
        lower, upper = self._safe_hsv_bounds(params["hsv_lower"], params["hsv_upper"], threshold_error, hsv_bonus)
        rectangular_mask = cv2.inRange(img_color_space, lower, upper)
        h_ref = hsv_mean[0]
        s_ref = hsv_mean[1]
        v_ref = hsv_mean[2]
        
        h_std = hsv_std[0] / 3.0
        s_std = hsv_std[1] / 3.0
        v_std = hsv_std[2] / 3.0

        # 计算向量距离阈值 tau
        tau_scale = params.get("tau_scale", 0.5)  # 缩小tau_scale，使阈值更严格

        tau = tau_scale * np.sqrt(0.8**2 * (h_std**2 + s_std**2 + v_std**2))
        
        # 构建矢量距离掩码
        h, s, v = cv2.split(img_color_space)
        # 计算欧式距离 (使用HSV循环距离)
        h_dist = np.minimum(np.abs(h.astype(np.int32) - h_ref), 180 - np.abs(h.astype(np.int32) - h_ref))
        s_dist = np.abs(s.astype(np.int32) - s_ref)
        v_dist = np.abs(v.astype(np.int32) - v_ref)
        
        # 缩放权重以平衡各通道贡献
        h_weight = 1.0 / max(1, h_std)
        s_weight = 1.0 / max(1, s_std)
        v_weight = 1.0 / max(1, v_std)
        
        # 计算加权欧氏距离
        distance = np.sqrt(
            (h_weight * h_dist)**2 + 
            (s_weight * s_dist)**2 + 
            (v_weight * v_dist)**2
        )
        
        vector_mask = np.zeros_like(h, dtype=np.uint8)
        vector_mask[distance <= tau] = 255
        from matplotlib import pyplot as plt
        #fig, axes = plt.subplots(2, 3, figsize=(10, 5))
        combined_mask = cv2.bitwise_or(rectangular_mask, vector_mask)
        original_combined_mask = combined_mask.copy()
        
        #axes[0, 0].imshow(combined_mask, cmap='gray')
        #axes[0, 0].set_title(f"Combined Mask {line_id}")
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small)
        #axes[0, 1].imshow(combined_mask, cmap='gray')
        #axes[0, 1].set_title(f"Morphology Open {line_id}")

        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_medium)
        #axes[0, 2].imshow(combined_mask, cmap='gray')
        #axes[0, 2].set_title(f"Morphology Close {line_id}")
        before_count = np.count_nonzero(original_combined_mask)
        after_count = np.count_nonzero(combined_mask)
        

        if after_count < before_count * 0.05 and before_count > 0:
            self.logger.info(f"Too strict for line {line_id}, using original mask")
            # 采用更保守的形态学处理
            combined_mask = original_combined_mask.copy()
            # 使用更小的内核进行开运算
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small)
            # 使用适中的内核进行闭运算
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_small)
        
        # 最后，进行轻微膨胀，以确保检测区域的完整性
        combined_mask = cv2.dilate(combined_mask, kernel_small, iterations=1)
        return combined_mask

    def _extract_boxes_from_mask(self, mask: np.ndarray, plot_count: bool = False) -> List[Tuple[int, int, int, int, float]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        height, width = mask.shape[:2]
        image_area = height * width
        dynamic_min_area = max(self.min_area, int(image_area * 0.001))  # at least 1% of the image
        dynamic_max_area = min(self.max_area, int(image_area * 0.05))   # at most 10% of the image

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < dynamic_min_area or area > dynamic_max_area:
                #self.logger.info(f"Area {area} is not in the range {dynamic_min_area} to {dynamic_max_area}")
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            #print(area)
            #print(x,y,w,h)
            #if plot_count:
            #    from matplotlib import pyplot as plt
            #    cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 0, 0), 3)
            #    plt.imshow(mask, cmap='gray')
            #    plt.show()

            
            aspect_ratio = w / h if h > 0 else 0
            if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                #self.logger.info(f"Aspect ratio {aspect_ratio} is not in the range {self.min_aspect_ratio} to {self.max_aspect_ratio}")
                continue
            
            # 计算圆形区域的填充度（
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            circle_area = np.pi * r * r
            circle_fill_ratio = area / circle_area if circle_area > 0 else 0
            
            # 使用圆形区域的填充度代替矩形区域的填充度
            if circle_fill_ratio < self.fill_ratio:
                #self.logger.info(f"Circle fill ratio {circle_fill_ratio} is less than {self.fill_ratio}")
                continue
            
            circularity = circle_fill_ratio  # 圆度就是圆形区域的填充度
            if circularity < 0.6:  
                # self.logger.info(f"Circularity {circularity} is less than 0.6")
                continue
            
            perimeter = cv2.arcLength(cnt, True)
            complexity = perimeter * perimeter / (4 * np.pi * area) if area > 0 else float('inf')
            if complexity > 3.0:  
                # self.logger.info(f"Complexity {complexity} is greater than 3.0")
                continue
            
            # 在置信度计算中，使用圆形区域的填充度
            confidence = (
                0.5 * circle_fill_ratio +                
                0.2 * (1 - abs(0.75 - aspect_ratio)) +  
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
    def _safe_hsv_bounds(self, lower_lst, upper_lst, thresh_err, hsv_bonus=(0,0,0)):
        """
        Generate safe HSV lower and upper bounds for color detection.
        
        This function ensures that the generated color bounds remain within valid HSV ranges:
        - H: [0, 180] (OpenCV's HSV format)
        - S: [0, 255]
        - V: [0, 255]
        
        It applies threshold errors and optional bonuses to expand or contract the range.
        
        Args:
            lower_lst: Initial lower bounds for HSV [H_low, S_low, V_low]
            upper_lst: Initial upper bounds for HSV [H_high, S_high, V_high]
            thresh_err: Error threshold to expand the range by
            hsv_bonus: Optional bonus values to further adjust each channel (H, S, V)
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Safe lower and upper bounds as uint8 arrays
        """

        lower = np.array(lower_lst, dtype=np.int16)
        upper = np.array(upper_lst, dtype=np.int16)
        lower -= (thresh_err + np.array(hsv_bonus))
        upper += (thresh_err + np.array(hsv_bonus))
        lower = np.clip(lower, (0,0,0), (180,255,255))
        upper = np.clip(upper, (0,0,0), (180,255,255))
        return lower.astype(np.uint8), upper.astype(np.uint8)
    def optimize_color_parameters(self, dataset: MetroDataset, logger=None, visualize=False) -> Dict[str, Any]:
        """
        Optimize color parameters based on training data.
        
        This function analyzes images from the dataset to extract dominant colors for each metro line.
        It processes images in the following steps:
        1. For each image in the dataset, extracts annotated regions
        2. Converts images to HSV color space via preprocessor
        3. Extracts dominant HSV values for each region
        4. Applies line-specific filtering rules to the color samples
        5. Computes statistics (mean, std) for each color channel
        6. Saves parameters for later use in detection
        
        Args:
            dataset: Dataset with ground truth annotations. Images should be in BGR uint8 format.
            logger: Optional logger for logging messages. If None, will create a new logger.
            visualize: Whether to visualize the dominant color extraction process. Helpful for debugging.
            
        Returns:
            Dict[str, Any]: Dictionary of optimized color parameters for each line:
                {
                    "line_id": {
                        "hsv_mean": Tuple[int, int, int],  # Mean H,S,V values
                        "hsv_std": Tuple[int, int, int],   # Standard deviation of H,S,V
                        "hsv_lower": List[int, int, int],  # Lower bound for color detection
                        "hsv_upper": List[int, int, int]   # Upper bound for color detection
                    }
                }
        
        Raises:
            ValueError: If the preprocessor is not correctly set
        """
        import cv2
        import numpy as np
        
        logger = logger or get_logger(__name__)
        logger.info("Starting color parameter optimization...")
        
        optimized_params = {}
        hsv_samples = {}
        
        # 采集样本阶段
        for idx in range(len(dataset)):
            image, annotations, _ = dataset.get_image_with_annotations(idx)
            if image.dtype == np.float32 and image.max() <= 1.0:
                image_cv = (image * 255).astype(np.uint8)
            else:
                image_cv = image.astype(np.uint8)
            
            if len(image_cv.shape) == 3 and image_cv.shape[2] == 3:
                bgr = image_cv
                if isinstance(self.preprocessor, ROIParamOptimizerPreprocessor):
                    hsv = self.preprocessor.preprocess(image_cv)
                else:
                    raise ValueError(f"Preprocessor {self.preprocessor} is not a ROIParamOptimizerPreprocessor")
            else:
                continue
                
            for annotation in annotations:
                x1, y1, x2, y2 = annotation[:4]
                roi_hsv = hsv[y1:y2, x1:x2]
                roi_bgr = bgr[y1:y2, x1:x2]
                line_id = annotation[4]
                try:
                    dominant_hsv = self.extract_dominant(roi_hsv, K=3)
                    hsv_samples.setdefault(line_id, []).append(dominant_hsv)
                    if visualize:
                        visualize_dominant_color(image, roi_bgr, dominant_hsv, line_id, x1, y1, x2, y2)
                except Exception as e:
                    logger.error(f"Error processing ROI for line {line_id}: {e}")
        
        # 处理样本阶段 - 过滤并计算参数
        for line_id, samples in hsv_samples.items():
            if not samples:
                logger.warning(f"No samples for line {line_id}, skipping")
                continue
            
            samples_array = np.array(samples)
            
            # 直接应用经验性过滤规则
            original_count = len(samples_array)
            filtered_array = samples_array.copy()
            if line_id in [12,6]:
                s_filter = samples_array[:, 1] <= 40
                if np.any(s_filter):
                    filtered_array = samples_array[s_filter]
                    logger.info(f"Line {line_id}: Filtered out {original_count - len(filtered_array)} samples with S>100")
                else:
                    logger.warning(f"Line {line_id}: All samples have S>100, keeping original samples")
                    
            # 应用特定线路的过滤规则
            if line_id in [ 12]:
                # 这些线路应剔除所有H>100的样本
                h_filter = samples_array[:, 0] <= 100
                if np.any(h_filter):  # 确保过滤后还有样本
                    filtered_array = samples_array[h_filter]
                    logger.info(f"Line {line_id}: Filtered out {original_count - len(filtered_array)} samples with H>100")
                else:
                    logger.warning(f"Line {line_id}: All samples have H>100, keeping original samples")
                    

            # 确保我们至少有一些样本用于计算统计量
            if len(filtered_array) == 0:
                logger.warning(f"No samples left for line {line_id} after filtering, using original samples")
                filtered_array = samples_array
            
            # 计算HSV统计量
            h_values = filtered_array[:, 0]
            avg_h = self._circular_mean_deg(h_values)
            avg_s = np.mean(filtered_array[:, 1])
            avg_v = np.mean(filtered_array[:, 2])
            
            std_h = self._circular_std_deg(h_values) 
            std_s = np.std(filtered_array[:, 1])
            std_v = np.std(filtered_array[:, 2])
            
            # 存储颜色参数
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
    def extract_dominant(self, roi: np.ndarray, K: int = 3, attempts: int = 10) -> Tuple[int, int, int]:
        """
        Extract dominant color from a single ROI using k-means clustering.
        
        This function analyzes the color distribution in an ROI to find the 
        most representative color. It uses k-means clustering on the pixels to
        group similar colors, then selects the cluster with the most pixels.
        
        Args:
            roi: Region of interest in HSV uint8 format. Expected to be a 3D array
                with shape (h, w, 3) representing an image region in HSV color space.
            K: Number of color clusters to create. Default is 3, which typically
               captures the main colors while filtering out noise.
            attempts: Number of times k-means algorithm will be restarted with
                     different centroid seeds. Higher values give better results
                     but take longer to compute.
        
        Returns:
            Tuple[int, int, int]: The dominant color in the same format as input (HSV).
                                 Values are integers in the range:
                                 - H: [0, 180]
                                 - S: [0, 255]
                                 - V: [0, 255]
        
        Raises:
            ValueError: If the ROI is empty or has invalid dimensions.
        """

        pixels = roi.reshape(-1, 3).astype(np.float32) 

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
    def _circular_mean_deg(self,degrees: np.ndarray) -> float:
        """
        Calculate circular mean of HSV hue values in degrees.
        
        This function properly handles the circular nature of HSV hue channel,
        where 0 and 180 degrees are adjacent. Standard mean calculation would
        produce incorrect results for values that cross the 0/180 boundary.
        
        Args:
            degrees: Array of hue values in degrees (range [0, 180])
            
        Returns:
            float: Circular mean of the hue values in range [0, 180]
        """
        radians = np.deg2rad(degrees * 2)  # HSV hue is from 0–180, so double for full circle
        sin_sum = np.sum(np.sin(radians))
        cos_sum = np.sum(np.cos(radians))
        mean_rad = np.arctan2(sin_sum, cos_sum)
        mean_deg = np.rad2deg(mean_rad) / 2
        if mean_deg < 0:
            mean_deg += 180
        return mean_deg
    def _circular_std_deg(self,degrees: np.ndarray) -> float:
        """
        Calculate circular standard deviation of HSV hue values in degrees.
        
        This function calculates the standard deviation in a way that respects the
        circular nature of HSV hue channel. It is based on the concentration
        parameter (R) of the circular distribution.
        
        Args:
            degrees: Array of hue values in degrees (range [0, 180])
            
        Returns:
            float: Circular standard deviation of hue values in degrees
        """
        radians = np.deg2rad(degrees * 2)
        sin_sum = np.sum(np.sin(radians))
        cos_sum = np.sum(np.cos(radians))
        R = np.sqrt(sin_sum**2 + cos_sum**2) / len(degrees)
        std_rad = np.sqrt(-2 * np.log(R))
        std_deg = np.rad2deg(std_rad) / 2  # back to HSV hue scale
        return std_deg
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
        #img_preprocessed = cv2.medianBlur(img_preprocessed, ksize=3)
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
    contours = detector._extract_boxes_from_mask(finial_mask, plot_count=True)
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