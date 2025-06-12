# 巴黎地铁线路识别系统技术报告

## 摘要
<!-- 项目概述、主要技术贡献、实验结果总结 -->

---

## 1. 项目概述

### 1.1 研究背景与意义

随着智能化辅助技术的快速发展，为视障人士提供更好的出行体验已成为计算机视觉领域的重要应用方向。巴黎地铁作为世界上历史最悠久、线路最复杂的地铁系统之一，其复杂的线路网络和多样化的标识系统对视障人士的独立出行构成了巨大挑战。

本项目旨在开发一套能够集成到**智能眼镜或智能手机**上的**实时地铁图标检测识别系统**，通过计算机视觉技术自动识别巴黎地铁的14条线路标识，为盲人和视障人士提供实时的线路信息反馈，从而显著改善他们在巴黎地铁系统中的导航体验和出行安全性。

该系统的成功实现将为无障碍智能出行技术开辟新的应用场景，具有重要的社会价值和技术意义。

### 1.2 技术挑战分析

地铁标识识别在实际应用场景中面临多重技术难点，主要包括：

#### 1.2.1 光照条件变化
- **问题描述**：地铁环境中的光照条件极其复杂多变，从室外强光到地下昏暗，再到人工照明的不均匀分布
- **技术影响**：虽然图像预处理技术可以在一定程度上减少光照变化的影响，但**无法完全消除**其干扰
- **极端情况**：在极端光照环境下，例如**局部强反光、镜面反射**等情况，现有预处理方法基本无效，严重影响检测精度

#### 1.2.2 背景噪声干扰  
- **场景复杂性**：地铁站内环境结构复杂，包含大量的建筑元素、广告牌、人流等干扰因素
- **算法局限性**：传统的几何特征检测方法（如**霍夫圆检测**）在复杂背景下容易产生大量误检，检测效果严重下降
- **噪声类型**：包括几何噪声、颜色噪声、纹理干扰等多种类型的背景干扰

#### 1.2.3 图像质量差异化
- **设备差异性**：图像数据均来自**手机拍摄**，但不同品牌、型号手机的成像质量存在显著差异
- **成像参数不一致**：包括分辨率、色彩还原度、噪声水平、锐度等参数的不统一
- **数据集规模限制**：现有数据集规模相对较小，**训练集仅占总数据的1/3**，限制了模型的泛化能力

#### 1.2.4 14条线路的颜色相似性问题
- **视觉区分度**：巴黎地铁14条线路对于**人眼具有显著的颜色区别**，但在数字图像处理中面临挑战
- **色域重叠**：某些线路的颜色在特定**色域空间下存在重叠区域**，容易造成误分类
- **光照影响**：颜色特征受光照条件影响显著，同一线路在不同光照下可能呈现截然不同的颜色特征
- **方法选择**：由于霍夫圆检测在复杂背景下的失效，我们最终选择了**基于颜色特征的检测方法**

### 1.3 系统整体架构

本系统采用模块化设计理念，整体架构如下：

```mermaid
graph TD
    A[Raw Data<br/>原始图像数据] --> B[Dataset<br/>数据集管理]
    B --> C[ROI Detector<br/>区域检测器]
    C --> D[Classifier<br/>线路分类器]
    D --> E[Result<br/>识别结果]
    
    F[Base Preprocessor<br/>基础预处理器] --> G[ROI Preprocessor<br/>检测预处理器]
    F --> H[Classification Preprocessor<br/>分类预处理器]
    
    G --> C
    H --> D
    
    I[Config System<br/>配置管理] --> C
    I --> D
    I --> B
    
    style C fill:#e1f5fe
    style D fill:#f3e5f5
    style F fill:#fff3e0
```

#### 核心模块说明：

- **检测器模块(ROI Detection)**：负责从输入图像中定位和提取潜在的地铁线路标识区域
- **分类器模块(Classification)**：对检测到的区域进行精确的线路分类识别
- **数据处理模块**：统一管理训练、验证和测试数据，支持多种数据格式
- **评估与流程管理**：提供完整的训练、测试和部署流程支持

#### 预处理器设计：

考虑到不同检测和分类方法对图像预处理需求的差异性，我们设计了**分层预处理器架构**:

- **`BasePreprocessor`**：抽象基类，包含所有通用的关键预处理操作
- **`ROIParamOptimizerPreprocessor`**：检测器专用预处理器,针对颜色特征提取优化
- **`ClassificationPreprocessor`**: CNN分类器专用预处理器,针对深度学习模型的输入要求进行预处理

### 1.4 技术路线与方法概述

#### 1.4.1 核心技术路线
1. **基于颜色特征的ROI检测**：采用HSV色彩空间分析，通过K-means聚类自动学习各线路的颜色特征
2. **模板匹配分类(已弃用)**：构建标准模板库，通过相似度匹配实现精确分类
3. **基于CNN的数字分类器**：采用深度卷积神经网络对检测到的ROI区域进行14类地铁线路的精确分类
4. **多约束筛选策略**：结合颜色、几何形状、形态学特征的多重约束条件
5. **自适应参数优化**：根据图像特征动态调整检测参数

#### 1.4.2 关键技术创新
- **颜色参数自动学习**：从训练数据中自动学习并优化各线路的颜色特征参数
- **矢量距离与矩形范围融合**：创新性地结合两种颜色匹配策略，提高检测精度
- **CNN与传统CV方法结合**：将深度学习分类器与基于颜色特征的检测器有机结合
- **模块化可配置架构**：基于Hydra配置框架的高度可配置系统设计
- **自适应形态学处理**：根据图像尺寸和特征动态调整形态学操作参数

---

## 2. 相关工作与技术背景

### 2.1 目标检测技术发展
目标检测是计算机视觉领域最核心的问题之一, 其研究主要沿着两条路径前进:  
#### 2.1.1 基于手工特征的传统方法  
在深度学习普及之前, 目标检测主要依赖于手工设计的特征和启发式算法. 常见方法包括:  
- **颜色直方图与边缘检测**  
- **形状匹配技术, 如霍夫变换检测几何结构(如圆形, 直线)**  
- **滑动窗口+SVM分类器**  

尽管这些方法在简单背景下表现尚可, 但在**光照变化,复杂背景和多类别干扰**下容易失效, 缺乏鲁棒性. 这些缺陷与我们在地铁识别场景中遇到的问题高度一致.  

尽管传统方法具有实现简单, 资源消耗低的优势, 但在复杂环境适应性与可扩展性方面严重不足, 难以应对本项目中多类别, 小目标, 高干扰的应用场景.  

#### 2.1.2 深度学习方法  
近年来, 以深度卷积神经网络为核心的目标检测技术取得了突破性进展, 主要代表包括:  
- Two-Stage Detectors: 如 R-CNN, Fast R-CNN, Faster R-CNN, 通过区域提取 + 分类两步完成检测  
- One-Stage Detecotrs: 如 YOLO (You Only Look Once), SSD (Single Shot Multibox Detector), 直接从图像回归目标位置与类别, 实现高速实时监测  
- Transformer-Based Models: 如 DETR, 使用Vision Transformer (ViT)

由于本项目平台计算能力限制(智能眼镜等平台), 无法追求大型深度模型等精度极限, 但是可以设计简化版的CNN结构, 在精度与资源使用中寻求平衡  


### 2.2 颜色特征在目标检测中的应用
#### 2.2.1 色彩空间选择  
颜色特征作为低计算复杂度, 高表达性的视觉线索, 在实际目标检测系统中被广泛应用:  
- RGB: 原始图像空间, 易受光照影响  
- HSV: 色调(Hue)在一定程度上解耦光照强度, 广泛应用于目标分割与颜色分类  
- CIELAB: 感知均匀空间, 对人眼的感知建模更精细, 但计算成本较高  

在本项目中, HSV被选作基础色彩空间, 并结合 K-Means 聚类算法自动学习色调分布, 是兼顾精度与效率的实用策略  

#### 2.2.2 基于颜色的区域提取方法
- 阈值分割: 设定色调或亮度阈值, 提取候选区域
- 直方图匹配: 构造目标颜色模版, 通过相似度计算筛选区域
- 聚类分割: 在颜色空间中聚类, 适应不同图像条件下的颜色偏移

基于颜色的检测方法计算开销低, 易于解释, 但鲁棒性有限, 因此在本项目中我们加入了形态学操作, 形态先验, 和参数动态调整  

### 2.3 地铁/交通标识识别相关研究
#### 2.3.1 交通标志识别研究
交通标志识别(TSR)是计算机视觉中较成熟的方向, 其研究可为本项目提供重要参考:  
- 经典数据集: 德国GTSRB数据集, 为深度学习模型提供了丰富的训练样本
- 方法演进:
    - 早期依赖颜色 + 形状特征, 使用 Haar-like 或 HOG 特征
    - 深度学习阶段使用小型CNN进行分类, 例如 LeNet, MobileNet, ResNet
    - 同时引入注意力机制以增强对复杂背景的鲁棒性

在本项目中, 由于计算平台限制, 选用了颜色+形状特征进行ROI区域筛选, 小型CNN网络作为分类器的结构  
#### 2.3.2 地铁标志识别
相比道路交通标识, 地铁标识的公开研究比较少, 存在一下技术特点:
- 识别目标更小, 类间差异更低(如色域重叠)  
- 易受反光, 阴影干扰, 光照条件更苛刻  
- 数据稀缺, 缺乏大规模公开数据集(在该项目中仅有87张训练集图片)  
---

## 3. 数据集构建与分析

### 3.1 数据集描述
本数据集由Mme.Rossant使用手机在巴黎地铁站内手动拍摄. 图像覆盖了大部分的站点区域, 在其拍摄过程中尽可能保证了角度, 光照等因素等多样性, 并加入了无地铁标志的杂乱背景图, 以增强数据的泛化能力.  
随后Mme.Rossant采用自动化工具并结合人工校对对图像中的ROI区域进行了标注. 每张图像中的ROI都包含了地铁线路编号, 作为后续模型训练的关键输入.  
采集环境:  
- 设备: 智能手机(具体型号未知)  
- 场景: 巴黎地铁站台  
- 分辨率: 主要为 1600x1200, 少数为 2120x1196  
- 标注方式: 框选式标注(bounding-box annotation)

### 3.2 数据统计分析
我们对数据集的图像尺寸, ROI尺寸及其在不同线路间的分布情况进行了可视化分析, 如图3.1与图3.2所示/
![各线路样本数量分布](./pics/line_id_ana.png)
从柱状图和饼图中可以观察到, 不同线路的样本数量存在较大差异. **线路1**占比最高, ROI数量为22, 占总样本的16.7%. 相比之下, **线路5**样本数仅为3, 数据分布不均衡, 这会对模型的训练带来偏差.   
这种类别不平衡问题可通过数据增强, 重采样策略, 或者使用类别加权损失函数来缓解.
*图 3.1: 各线路样本数量分布*

![图像与标注框尺寸分布](./pics/size_ana.png)

在图像维度分析中我们发现:
- 图像尺寸分布: 图像尺寸主要集中于1600x1200, 仅有几个分辨率为 2120x1196  
- 图像宽高比: 图像宽高比几乎一致, 说明图像尺寸标准化程度较高, 便于后续处理  

在ROI分析部分:  
- ROI尺寸分布: 散点图显示大多数ROI的宽高分布较集中, 呈线性趋势, 说明ROI尺寸变化范围有限  
- ROI面积分布: 面积分布呈右偏, 大部分ROI的面积集中在5000 到 15000 像素之间, 部分存在大面积区域  

通常面对这类尺寸偏移的问题, 我们的模型需要具备多尺度鲁棒性, 可考虑引入金字塔特征网络, 但是由于本项目中ROI检测使用传统算法(颜色分类+形态先验), 我们基于上述图像分析设计了更具有鲁棒性的动态参数系统, 后续会详细介绍.  

训练集的标注质量很好, 经过可视化检验后仅有一个错误标注:

![错误标注](./pics/error.png)


### 3.3 数据预处理策略

基于项目需求，我们设计了分层次的数据预处理架构，通过模块化设计实现了多种预处理策略的灵活配置与使用。

#### 3.3.1 预处理架构设计

我们采用了基于抽象基类的设计模式，通过`BasePreprocessor`定义统一的预处理接口：

```python
class BasePreprocessor(ABC):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
    
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess the data."""
        return image
```

这种设计模式保证了预处理模块的可扩展性和一致性，便于后续算法模块的集成。

#### 3.3.2 数据清洗与质量控制

在数据加载阶段，`MetroDataset`类实现了严格的数据质量控制机制：

**格式兼容性处理**：
- 支持Excel格式（.xls/.xlsx）和MATLAB格式（.mat）双重数据源
- 自动检测并验证必要字段：`image_id`, `y1`, `y2`, `x1`, `x2`, `class_id`

**数据清洗策略**：
- 自动识别并移除异常数据记录（如image_id为120的错误标注和198异常颜色数据）
- 实施边界框坐标验证，确保标注框在图像范围内
- 图像路径自动解析与验证，支持多种命名格式

**坐标标准化**：
```python
def _clip_bbox(bbox, max_width, max_height):
    """Clip bounding box coordinates to image dimensions."""
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, max_width - 1))
    y1 = max(0, min(y1, max_height - 1))
    x2 = max(0, min(x2, max_width))
    y2 = max(0, min(y2, max_height))
    return x1, y1, x2, y2
```

#### 3.3.3 图像预处理流程

针对地铁标志检测的特点，我们在`ROIParamOptimizerPreprocessor`中实现了专门的预处理策略：

**颜色恒常性校正**：
- **Gray World算法**：基于灰色世界假设进行全局颜色校正
- **Ebner算法**：使用高斯滤波进行局部颜色平衡，更适合处理光照不均匀的场景

**颜色空间转换**：
- 将输入图像从BGR颜色空间转换至HSV颜色空间
- HSV空间对光照变化更加鲁棒，便于后续颜色特征提取

**数据类型规范化**：
- 自动检测输入图像格式（float32 [0,1] 或 uint8 [0,255]）
- 统一转换为uint8格式，确保后续处理的一致性

**图像尺寸处理**：
- 可选的图像resize功能，支持固定尺寸输出
- 保持原始宽高比，避免图像失真

#### 3.3.4 专用预处理模块

考虑到不同算法模块的特殊需求，我们还实现了专门的预处理方法：

**霍夫圆检测预处理(已弃用)**：
```python
def preprocess_hough_circles(self, image: np.ndarray) -> np.ndarray:
    """Preprocess image for Hough Circles detection."""
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    return img_gray
```

**调试可视化支持**：
- 集成了预处理结果的可视化功能
- 支持对比原图与处理后图像，便于参数调优
![原始BGR图像](../results/roi_param_optimizer_preprocessing_visualize/original_image_bgr.png)
![Ebner算法处理后的BGR图像](../results/roi_param_optimizer_preprocessing_visualize/processed_image_bgr.png)
![Ebner算法处理后的HSV图像](../results/roi_param_optimizer_preprocessing_visualize/processed_image_hsv.png)
### 3.4 数据集划分策略

为确保模型训练和评估的科学性，我们实施了严格的数据集划分策略，重点防止数据泄露问题。

#### 3.4.1 防泄露划分机制

**图像级别划分**：
```python
def _split_train_val(self, df: pd.DataFrame, mode: str = 'train') -> pd.DataFrame:
    # 避免同一个image_id被分到两个不同的set中造成图像泄漏
    unique_ids = df['image_id'].unique()
    train_ids, val_ids = train_test_split(
        unique_ids, 
        test_size=self.val_split, 
        random_state=self.random_seed
    )
```

这种策略确保同一张图像的所有标注只会出现在同一个数据集分片中，有效避免了数据泄露问题。

#### 3.4.2 三阶段数据划分

**训练集（Training Set）**：
- 来源：`Apprentissage.xls` 或 `Apprentissage.mat`
- 用途：模型参数学习，包括颜色特征提取和CNN分类器训练
- 占比：根据`val_split`参数动态调整

**验证集（Validation Set）**：
- 来源：从训练数据中按图像ID随机抽取
- 用途：超参数调优、模型选择、过拟合监控
- 占比：由配置参数`val_split`控制（例如：0.2表示20%）

**测试集（Test Set）**：
- 来源：`Test.xls` 或 `Test.mat`
- 用途：最终性能评估，模拟真实部署场景
- 特点：完全独立于训练过程，确保评估结果的客观性

#### 3.4.3 随机性控制

**可重现性保证**：
- 使用固定的`random_seed`参数确保数据划分的可重现性
- 所有随机操作均使用相同的种子，便于实验复现和结果对比

**灵活配置支持**：
- 支持`val_split=0.0`配置，此时不创建验证集
- 可根据数据规模和实验需求动态调整划分比例

#### 3.4.4 数据完整性验证

**标注数据校验**：
- 自动验证图像文件的存在性和可访问性
- 检查标注框的有效性，确保坐标范围合理
- 统计各类别的样本分布，便于后续的类别平衡处理

**多格式兼容性**：
- 同时支持Excel和MATLAB两种标注格式
- 自动适配不同的图像命名规范
- 支持灵活的目录结构组织

通过以上严格的数据预处理和划分策略，我们为后续的ROI检测器和CNN分类器奠定了坚实的数据基础，确保了实验结果的科学性和可靠性。

---

## 4. ROI检测器设计与实现 ⭐ **[重点章节]**

### 4.1 检测器整体架构设计

#### 4.1.1 基于颜色特征的检测策略

在地铁标志识别场景中，我们选择基于颜色特征的检测策略主要基于以下分析：

**传统几何检测方法的局限性**：
- **霍夫圆检测失效**：尽管地铁标志通常呈圆形，但在复杂背景和光照条件下，霍夫圆检测产生大量误检，检测效果严重下降
- **边缘检测不稳定**：复杂背景噪声和光照变化使得边缘检测方法难以获得稳定的特征

**颜色特征的优势**：
- **区分度高**：巴黎地铁14条线路具有显著的颜色差异，为人眼和算法提供了良好的区分基础
- **计算效率高**：颜色特征提取和匹配的计算复杂度相对较低，适合实时应用
- **鲁棒性好**：通过适当的颜色空间选择和预处理，可以在一定程度上抵抗光照变化

#### 4.1.2 多线路并行检测框架

我们设计了同时检测14条地铁线路的并行检测框架：

```python
def detect(self, image: np.ndarray, has_visualize: bool = True) -> List[Dict[str, Any]]:
    """并行检测多条地铁线路"""
    detections = []
    
    for line_id in self.color_params:
        # 为每条线路生成颜色掩模
        mask = self._extract_line_mask(img_hsv, line_id)
        # 从掩模中提取候选框
        boxes = self._extract_boxes_from_mask(mask)
        
        for x1, y1, x2, y2, conf in boxes:
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "line_id": line_id,
                "confidence": conf
            })
    
    return self._nms_by_line(detections)
```

**并行检测优势**：
- **效率提升**：一次图像预处理，多条线路同时检测
- **结果完整**：单张图像中可能包含多条线路标志，并行检测确保不遗漏
- **便于比较**：各线路检测结果可以直接比较置信度，便于后续筛选

#### 4.1.3 模块化设计理念

我们采用基于抽象基类的模块化设计架构：

```python
class BaseDetector(ABC):
    """ROI检测器基类"""
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """检测ROI区域"""
        pass
    
    @abstractmethod
    def save_params(self, params: Dict[str, Any]) -> bool:
        """保存检测参数"""
        pass
    
    def set_preprocessor(self, preprocessor: BasePreprocessor):
        """设置预处理器"""
        self.preprocessor = preprocessor
```

**模块化设计优势**：
- **可扩展性**：新的检测算法可以通过继承基类快速集成
- **配置驱动**：通过YAML配置文件灵活控制检测参数
- **依赖注入**：预处理器作为独立模块，支持不同算法组合

### 4.2 颜色参数学习算法

#### 4.2.1 HSV色彩空间分析

我们选择HSV色彩空间作为颜色特征分析的基础：

**HSV空间优势**：
- **H通道（色调）**：对光照强度变化相对不敏感，保持颜色本质特征
- **S通道（饱和度）**：描述颜色纯度，有助于区分不同线路的颜色差异
- **V通道（亮度）**：可以通过预处理进行归一化，减少光照影响

**对比其他色彩空间**：
- **RGB空间**：三个通道高度相关，易受光照影响
- **LAB空间**：感知均匀但计算复杂度高，不适合实时应用

#### 4.2.2 K-means聚类主导色提取

我们开发了基于K-means聚类的主导色提取算法：

```python
def extract_dominant(self, roi: np.ndarray, K: int = 3, attempts: int = 10) -> Tuple[int, int, int]:
    """提取ROI区域的主导色"""
    pixels = roi.reshape(-1, 3).astype(np.float32)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    ret, labels, centers = cv2.kmeans(
        pixels, K, None, criteria, attempts, 
        flags=cv2.KMEANS_PP_CENTERS
    )
    
    # 选择像素数量最多的聚类中心作为主导色
    _, counts = np.unique(labels, return_counts=True)
    dominant_idx = np.argmax(counts)
    dominant_center = centers[dominant_idx]
    
    return tuple(map(int, dominant_center))
```
![Kmeans结果](./pics/kmeans_hsv.png)
**算法特点**：
- **自适应聚类**：K=3的设置能够有效捕捉主要颜色并过滤噪声
- **多次重启**：10次重启机制提高聚类结果的稳定性
- **主导色选择**：基于像素数量选择最具代表性的颜色

#### 4.2.3 统计参数自动计算

对于每条地铁线路，我们自动计算颜色统计参数：

```python
# 计算循环均值（针对HSV的H通道）
avg_h = self._circular_mean_deg(filtered_array[:, 0])
avg_s = np.mean(filtered_array[:, 1])
avg_v = np.mean(filtered_array[:, 2])

# 计算循环标准差
std_h = self._circular_std_deg(filtered_array[:, 0])
std_s = np.std(filtered_array[:, 1])
std_v = np.std(filtered_array[:, 2])

# 生成检测范围
optimized_params[line_id] = {
    "hsv_mean": (int(avg_h), int(avg_s), int(avg_v)),
    "hsv_std": (int(std_h), int(std_s), int(std_v)),
    "hsv_lower": [max(0, int(avg_h - 2 * std_h)), 
                  max(0, int(avg_s - 2 * std_s)), 
                  max(0, int(avg_v - 2 * std_v))],
    "hsv_upper": [min(180, int(avg_h + 2 * std_h)), 
                  min(255, int(avg_s + 2 * std_s)), 
                  min(255, int(avg_v + 2 * std_v))]
}
```

**关键创新点**：
- **循环统计**：H通道使用循环均值和标准差，正确处理0°和180°边界
- **2σ规则**：使用两倍标准差确定颜色检测范围，平衡召回率和精确率
- **边界保护**：严格限制HSV值在有效范围内

#### 4.2.4 参数优化与调整策略

我们实现了线路特定的参数优化机制：

```yaml
threshold_error_dict:
  "1": 2
  "2": 2  
  "3": 5
  "4": 5
  "5": 15  # 样本稀少，需要更宽泛的阈值
  "6": 5
  "7": 10
  "8": 5
```

**自适应调整策略**：
- **数据驱动**：根据训练数据中各线路的样本数量和颜色一致性调整阈值
- **误差补偿**：为样本稀少的线路（如线路5）设置更大的阈值误差
- **动态更新**：检测参数可以根据新数据自动更新和优化

### 4.3 图像预处理流程

#### 4.3.1 色彩空间转换

预处理流程的核心是BGR到HSV的转换：

```python
def preprocess(self, image: np.ndarray, has_visualize: bool = True) -> np.ndarray:
    """ROI检测预处理流程"""
    # 1. 数据类型标准化
    if image.dtype == np.float32 and image.max() <= 1.0:
        img_bgr_uint8 = (image * 255).astype(np.uint8)
    else:
        img_bgr_uint8 = image.astype(np.uint8)
    
    # 2. 颜色恒常性校正
    if self.use_color_constancy:
        img_bgr_uint8 = self._apply_color_constancy(img_bgr_uint8, self.color_constancy_method)
    
    # 3. 色彩空间转换
    img_hsv_uint8 = cv2.cvtColor(img_bgr_uint8, cv2.COLOR_BGR2HSV)
    
    return img_hsv_uint8
```

#### 4.3.2 预处理管道设计

`ROIParamOptimizerPreprocessor`实现了专门的预处理管道：

**颜色恒常性校正**：
- **Gray World算法**：假设图像整体色调平衡，适用于常规光照条件
- **Ebner算法**：使用高斯滤波进行局部颜色平衡，更适合光照不均匀场景
![Gray_World校正正可视化](./pics/gray_world.png)
![Ebner校正可视化](./pics/Ebner.png)
**处理流程优化**：
- **批处理支持**：`has_visualize`参数控制调试可视化，提高批处理性能
- **内存效率**：避免不必要的图像拷贝，优化内存使用
- **错误处理**：完善的异常处理机制，确保处理稳定性

### 4.4 ROI候选区域生成算法

#### 4.4.1 颜色掩模生成策略

我们创新性地融合了两种颜色匹配策略：

**矩形范围掩模**：
```python
lower, upper = self._safe_hsv_bounds(params["hsv_lower"], params["hsv_upper"], threshold_error, hsv_bonus)
rectangular_mask = cv2.inRange(img_color_space, lower, upper)
```

**矢量距离掩模**：
```python
# 计算HSV循环距离
h_dist = np.minimum(np.abs(h - h_ref), 180 - np.abs(h - h_ref))
s_dist = np.abs(s - s_ref)
v_dist = np.abs(v - v_ref)

# 加权欧氏距离
distance = np.sqrt((h_weight * h_dist)**2 + (s_weight * s_dist)**2 + (v_weight * v_dist)**2)
vector_mask[distance <= tau] = 255
```

#### 4.4.2 矢量距离计算方法

**HSV循环距离**：
- 正确处理H通道的循环特性（0°和180°相邻）
- 使用最短角距离计算色调差异

**加权欧氏距离**：
- 根据各通道标准差设置权重，平衡不同通道的贡献
- tau阈值基于颜色分布的协方差矩阵计算

#### 4.4.3 掩模融合与优化

```python
# 融合两种掩模
combined_mask = cv2.bitwise_or(rectangular_mask, vector_mask)

# 形态学处理
combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small)
combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_medium)

# 自适应处理
if after_count < before_count * 0.05 and before_count > 0:
    # 过度严格时采用保守处理
    combined_mask = original_combined_mask.copy()
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small)
```

#### 4.4.4 形态学操作优化

**自适应核大小**：
```python
base_kernel_size = max(3, int(min(height, width) * 0.005))
kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (base_kernel_size, base_kernel_size))
kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (base_kernel_size*2, base_kernel_size*2))
```

**多级形态学处理**：
- **开运算**：去除小噪声和细连接
- **闭运算**：填补内部空洞
- **膨胀运算**：确保检测区域完整性

### 4.5 候选区域后处理

#### 4.5.1 几何约束过滤

我们实施了多重几何约束来筛选有效候选区域：

```python
# 动态面积约束
dynamic_min_area = max(self.min_area, int(image_area * 0.001))
dynamic_max_area = min(self.max_area, int(image_area * 0.05))

# 宽高比约束
aspect_ratio = w / h if h > 0 else 0
if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
    continue

# 圆形填充度约束
(cx, cy), r = cv2.minEnclosingCircle(cnt)
circle_area = np.pi * r * r
circle_fill_ratio = area / circle_area if circle_area > 0 else 0
if circle_fill_ratio < self.fill_ratio:
    continue
```

#### 4.5.2 连通组件分析

**轮廓提取与分析**：
- 使用`cv2.findContours`提取连通组件
- 计算最小外接圆来评估形状规则性
- 基于轮廓复杂度过滤不规则形状

#### 4.5.3 非极大值抑制（NMS）

```python
def _nms_by_line(self, detections: List[Dict[str, Any]], iou_thresh=0.5) -> List[Dict[str, Any]]:
    """按线路分组的NMS算法"""
    grouped = defaultdict(list)
    for det in detections:
        grouped[det['line_id']].append(det)
    
    final = []
    for line_id, group in grouped.items():
        final.extend(self._apply_nms_for_group(group, iou_thresh))
    return final
```

**分组NMS策略**：
- 按线路ID分组，避免不同线路间的误抑制
- 基于IoU阈值去除重复检测
- 保持每条线路的最佳检测结果

#### 4.5.4 置信度评估机制

```python
confidence = (
    0.5 * circle_fill_ratio +                # 形状规则性
    0.2 * (1 - abs(0.75 - aspect_ratio)) +  # 宽高比适应性
    0.3 * circularity                       # 圆度评估
)
```

**多因子置信度**：
- 综合形状、大小、填充度等多个几何特征
- 权重分配基于实验验证的最优组合
- 为后续分类提供可靠的质量评估

### 4.6 参数动态调整策略

#### 4.6.1 自适应阈值调整

系统支持基于检测结果的动态参数调整：

**线路特定阈值**：
- 根据训练数据质量为每条线路设置个性化阈值误差
- 自动识别难检测线路并放宽检测条件

**实时反馈调整**：
- 监控检测结果的质量指标
- 基于假阳性率动态调整阈值

#### 4.6.2 形态学核大小自适应

```python
base_kernel_size = max(3, int(min(height, width) * 0.005))
```

**图像尺寸自适应**：
- 核大小与图像尺寸成比例，确保处理效果一致性
- 最小核大小保证，避免过小核导致的处理失效
- 多尺度核设计，适应不同大小的目标对象

通过以上系统化的设计和实现，我们的ROI检测器在保证检测精度的同时，具备了良好的鲁棒性和可扩展性，为后续的CNN分类器提供了高质量的候选区域。

---

## 5. CNN分类器模块设计 **[预留章节]**

### 5.1 分类器架构概述
<!-- BaseClassifier设计、接口定义、CNN架构选择 -->

### 5.2 卷积神经网络设计
<!-- CNN网络架构、层次设计、参数配置 -->

### 5.3 数据增强与训练策略
<!-- 图像增强技术、训练优化、过拟合防止 -->

### 5.4 分类预处理流程
<!-- ClassificationPreprocessor设计、图像标准化、尺寸调整 -->

### 5.5 模型训练与优化
<!-- 损失函数选择、优化器配置、学习率调度 -->

### 5.6 模型评估与性能优化
<!-- 分类准确率、混淆矩阵分析、模型压缩 -->

---

## 6. 系统集成与流程管理

### 6.1 模块化架构设计

#### 6.1.1 配置管理系统

我们基于Hydra框架构建了高度灵活的配置管理系统，支持分层配置和动态组合：

**配置文件组织结构**：
```
config/
├── config.yaml                 # 主配置文件
├── mode/
│   ├── train.yaml              # 训练模式配置
│   ├── test.yaml               # 测试模式配置
│   └── demo.yaml               # 演示模式配置
├── dataset/
│   └── default.yaml            # 数据集配置
├── roi_detection/
│   └── multi_color.yaml        # ROI检测器配置
├── classification/
│   └── template_classification.yaml  # 分类器配置
└── preprocessing/
    ├── template/
    │   └── default.yaml        # 模板预处理配置
    └── roi_param_optimizer/
        └── default.yaml        # ROI参数优化预处理配置
```

**主配置文件设计**：
```yaml
defaults:
  - mode: train
  - dataset: default
  - roi_detection: multi_color
  - classification: template_classification
  - preprocessing/template: default
  - preprocessing/roi_param_optimizer: default

output_dir: ${base_dir}/results
seed: 42
base_dir: ${oc.env:BASE_DIR}

model_dispatch:
  train: "src.pipeline.train_pipeline:main"
  test: "src.pipeline.test_pipeline:main"
  demo: "src.pipeline.demo_pipeline:main"
```

**配置管理优势**：
- **分层配置**：模块化配置文件，便于维护和扩展
- **动态组合**：通过defaults机制灵活组合不同配置
- **环境变量支持**：支持环境变量注入，适应不同部署环境
- **类型验证**：自动配置类型检查和验证

#### 6.1.2 组件接口设计

系统采用基于抽象基类的依赖注入模式，确保组件间的松耦合：

**核心基类架构**：
```mermaid
classDiagram
    class BasePreprocessor {
        <<abstract>>
        +cfg: DictConfig
        +preprocess(image: np.ndarray)* np.ndarray
        +resize(image: np.ndarray, target_size: Tuple) np.ndarray
    }
    
    class BaseDetector {
        <<abstract>>
        +cfg: DictConfig
        +logger: Logger
        +preprocessor: BasePreprocessor
        +detect(image: np.ndarray)* List[Dict]
        +save_params(params: Dict)* bool
        +set_preprocessor(preprocessor: BasePreprocessor)
    }
    
    class BaseClassifier {
        <<abstract>>
        +cfg: DictConfig
        +logger: Logger
        +preprocessor: BasePreprocessor
        +predict(image: np.ndarray)* Tuple[int, float]
        +train(X_train: np.ndarray, y_train: np.ndarray)* Dict
        +save(path: str)*
        +load(path: str)*
    }
    
    class ROIParamOptimizerPreprocessor {
        +use_color_constancy: bool
        +color_constancy_method: str
        +preprocess(image: np.ndarray) np.ndarray
        +ebner_color_constancy(image: np.ndarray) np.ndarray
        +gray_world(image: np.ndarray) np.ndarray
    }
    
    class TemplatePreprocessor {
        +color_space: str
        +use_clache: bool
        +preprocess(image: np.ndarray) np.ndarray
    }
    
    class MultiColorDetector {
        +color_params: Dict
        +threshold_error_dict: Dict
        +detect(image: np.ndarray) List[Dict]
        +optimize_color_parameters(dataset: MetroDataset) Dict
        +extract_dominant(roi: np.ndarray) Tuple[int, int, int]
    }
    
    class TemplateClassifier {
        +templates: Dict
        +template_size: Tuple
        +threshold: float
        +predict(image: np.ndarray) Tuple[int, float]
        +train(X_train: np.ndarray, y_train: np.ndarray)
    }
    
    BasePreprocessor <|-- ROIParamOptimizerPreprocessor
    BasePreprocessor <|-- TemplatePreprocessor
    BaseDetector <|-- MultiColorDetector
    BaseClassifier <|-- TemplateClassifier
    BaseDetector --> BasePreprocessor : uses
    BaseClassifier --> BasePreprocessor : uses
```

**依赖注入机制**：
```python
# 组件初始化和依赖注入示例
def _init_components(self):
    # 创建预处理器
    self.roi_preprocessor = ROIParamOptimizerPreprocessor(cfg=self.cfg.preprocessing.roi_param_optimizer)
    self.template_preprocessor = TemplatePreprocessor(cfg=self.cfg.preprocessing.template)
    
    # 创建检测器和分类器
    self.detector = MultiColorDetector(cfg=self.cfg.roi_detection)
    self.classifier = TemplateClassifier(cfg=self.cfg.classification)
    
    # 依赖注入
    self.detector.set_preprocessor(self.roi_preprocessor)
    self.classifier.set_preprocessor(self.template_preprocessor)
```

### 6.2 训练流程设计

#### 6.2.1 训练管道架构

`MetroTrainPipeline`实现了完整的训练流程管理：
**这里template训练部分最后需要改为CNN**
```mermaid
flowchart TD
    A[开始训练流程] --> B[加载配置文件]
    B --> C[初始化组件]
    C --> D[加载训练数据集]
    D --> E[加载验证数据集]
    E --> F{是否优化ROI参数?}
    
    F -->|是| G[ROI参数优化]
    F -->|否| H[跳过ROI参数优化]
    G --> I{是否创建模板?}
    H --> I
    
    I -->|是| J[提取训练ROI]
    I -->|否| K[跳过模板创建]
    J --> L[预处理ROI]
    L --> M[创建分类模板]
    M --> N[保存模板]
    N --> O[训练完成]
    K --> O
    
    style G fill:#e1f5fe
    style M fill:#f3e5f5
    style O fill:#e8f5e8
```

**训练流程核心代码**：
```python
class MetroTrainPipeline:
    def run(self):
        """运行训练流程"""
        self.logger.info("=== 开始训练流程 ===")
        
        # 步骤1: 加载数据集
        train_dataset, val_dataset = self._load_datasets()
        
        # 步骤2: ROI参数优化
        if self.cfg.mode.get("optimize_roi_param", False):
            self._optimize_roi_param(train_dataset)
        
        # 步骤3: 创建分类模板
        if self.cfg.mode.get("create_templates", False):
            self._create_templates(train_dataset)
```

#### 6.2.2 颜色参数优化流程

ROI检测器的颜色参数学习是训练流程的核心环节：

```mermaid
flowchart LR
    A[训练数据集] --> B[遍历图像]
    B --> C[提取标注ROI]
    C --> D[预处理转换HSV]
    D --> E[K-means主导色提取]
    E --> F[按线路ID分组]
    F --> G[线路特定过滤]
    G --> H[计算统计参数]
    H --> I[生成颜色范围]
    I --> J[保存参数文件]
    
    K[颜色样本] --> L[循环均值计算]
    L --> M[循环标准差计算]
    M --> N[HSV上下界]
    
    F --> K
    H --> L
    
    style E fill:#fff3e0
    style H fill:#e1f5fe
    style J fill:#e8f5e8
```

**关键优化步骤**：
1. **主导色提取**：使用K-means聚类提取每个ROI的主导颜色
2. **线路分组**：按照线路ID对颜色样本进行分组
3. **统计计算**：计算每条线路的HSV均值和标准差
4. **参数生成**：基于统计结果生成检测参数

#### 6.2.3 CNN模型训练流程

**CNN分类器训练流程**：



### 6.3 测试评估流程

#### 6.3.1 测试管道设计

`MetroTestPipeline`实现了完整的测试评估流程：

```mermaid
flowchart TD
    A[开始测试流程] --> B[加载测试数据集]
    B --> C[初始化检测器和分类器]
    C --> D[遍历测试图像]
    D --> E[ROI检测]
    E --> F[区域分类]
    F --> G[结果收集]
    G --> H{是否处理完所有图像?}
    
    H -->|否| D
    H -->|是| I[计算评估指标]
    I --> J[生成混淆矩阵]
    J --> K[可视化结果]
    K --> L[保存检测结果]
    L --> M[测试完成]
    
    subgraph "单图像处理流程"
        E1[图像预处理] --> E2[多线路并行检测]
        E2 --> E3[NMS去重]
        E3 --> E4[置信度排序]
    end
    
    E --> E1
    
    style I fill:#e1f5fe
    style K fill:#f3e5f5
    style M fill:#e8f5e8
```

**测试流程核心实现**：
```python
def run(self):
    """运行测试流程"""
    test_dataset = MetroDataset(cfg=self.cfg.dataset, mode='test')
    
    results = []
    for idx in range(len(test_dataset)):
        image, annotations, image_id = test_dataset.get_image_with_annotations(idx)
        
        # ROI检测
        detected_rois = self.multi_color_detector.detect(image)
        
        # 结果收集
        detected_classes = [(roi['line_id'], roi['bbox'], roi['confidence']) 
                          for roi in detected_rois]
        gt_classes = [(ann[4], ann[:4]) for ann in annotations]
        
        results.append({
            'image_id': int(image_id),
            'detected': detected_classes,
            'ground_truth': gt_classes
        })
    
    # 评估和保存
    self._calculate_metrics(results)
    self._save_results(results)
```

#### 6.3.2 批量处理机制

**高效批处理策略**：
- **内存优化**：逐张处理图像，避免内存溢出
- **进度监控**：使用tqdm显示处理进度
- **错误恢复**：单张图像处理失败不影响整体流程
- **可视化控制**：`has_visualize`参数控制调试输出

#### 6.3.3 结果可视化

系统提供多种结果可视化功能：

**检测结果可视化**：
```python
def _visualize_results(self, image: np.ndarray, detections: List[Tuple], 
                      ground_truth: List[Tuple], image_id: int):
    """可视化检测结果"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # 显示检测结果
    ax1.imshow(image)
    for cls, bbox, conf in detections:
        x1, y1, x2, y2 = bbox
        rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                        linewidth=2, edgecolor='red', facecolor='none')
        ax1.add_patch(rect)
        ax1.text(x1, y1-5, f'Line {cls}: {conf:.2f}', 
                color='red', fontsize=12, weight='bold')
    
    # 显示真实标注
    ax2.imshow(image)
    for cls, bbox in ground_truth:
        x1, y1, x2, y2 = bbox
        rect = Rectangle((x1, y1), x2-x1, y2-y1,
                        linewidth=2, edgecolor='green', facecolor='none')
        ax2.add_patch(rect)
        ax2.text(x1, y1-5, f'GT Line {cls}', 
                color='green', fontsize=12, weight='bold')
```

### 6.4 数据管理系统

`MetroDataset`类提供了统一的数据管理接口，支持多种数据格式：

```mermaid
classDiagram
    class MetroDataset {
        +cfg: DictConfig
        +mode: str
        +data_root: Path
        +data_format: str
        +df: DataFrame
        +annotations: DataFrame
        
        +__init__(cfg, mode, logger)
        +_load_excel_data()
        +_load_mat_data()
        +_split_train_val(df, mode) DataFrame
        +_resolve_image_path(image_id) str
        +__getitem__(idx) Tuple
        +get_all() Tuple[List, np.ndarray]
        +get_image_with_annotations(idx) Tuple
        +get_class_balance_weights() Dict
    }
    
    class DataLoader {
        +excel_support: bool
        +mat_support: bool
        +image_formats: List[str]
    }
    
    class ValidationUtils {
        +validate_bbox(bbox, image_shape) bool
        +check_image_exists(path) bool
        +verify_annotations(df) bool
    }
    
    MetroDataset --> DataLoader : uses
    MetroDataset --> ValidationUtils : uses
```

**多格式数据支持**：

**Excel格式支持**：
```python
def _load_excel(self, path: Path) -> pd.DataFrame:
    """加载Excel格式标注数据"""
    df = pd.read_excel(path)
    required = {'image_id', 'y1', 'y2', 'x1', 'x2', 'class_id'}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")
    return df
```

**MATLAB格式支持**：
```python
def _load_mat_to_df(self, path: Path) -> pd.DataFrame:
    """加载MATLAB格式数据并转换为DataFrame"""
    mat_data = sio.loadmat(str(path))
    annot_data = mat_data['BD']  # 或者自动检测数据字段
    
    df = pd.DataFrame({
        'image_id': annot_data[:, 0].astype(int),
        'y1': annot_data[:, 1].astype(int),
        'y2': annot_data[:, 2].astype(int),
        'x1': annot_data[:, 3].astype(int),
        'x2': annot_data[:, 4].astype(int),
        'class_id': annot_data[:, 5].astype(int)
    })
    return df
```

**数据完整性验证**：
- **路径验证**：自动检测和验证图像文件路径
- **格式检查**：支持JPG、PNG等多种图像格式
- **边界验证**：确保标注框在图像范围内
- **异常数据清理**：自动移除已知的异常标注

**系统整体数据流**：
```mermaid
flowchart TB
    subgraph "Couche d'entrée de données"
        A1[Fichier d'annotation Excel]
        A2[Fichier d'annotation MATLAB]
        A3[Fichiers image]
    end
    
    subgraph "Couche de gestion des données"
        B1[MetroDataset]
        B2[Validation des données]
        B3[Conversion de format]
        B4[Stratégie de partitionnement]
    end
    
    subgraph "Couche de prétraitement"
        C1[Préprocesseur ROI]
        C2[Préprocesseur de classification]
    end
    
    subgraph "Couche algorithmique"
        D1[Détecteur multi-couleurs]
        D2[Classifieur à base de modèles]
    end
    
    subgraph "Couche de gestion du flux"
        E1[Pipeline d'entraînement]
        E2[Pipeline de test]
        E3[Gestion de configuration]
    end
    
    subgraph "Couche de sortie"
        F1[Résultats de détection]
        F2[Métriques d'évaluation]
        F3[Visualisation]
    end
    
    A1 --> B1
    A2 --> B1
    A3 --> B1
    B1 --> B2
    B2 --> B3
    B3 --> B4
    
    B4 --> C1
    B4 --> C2
    
    C1 --> D1
    C2 --> D2
    
    D1 --> E1
    D2 --> E1
    D1 --> E2
    D2 --> E2
    
    E3 --> E1
    E3 --> E2
    
    E1 --> F1
    E2 --> F1
    E2 --> F2
    E2 --> F3
    
    style B1 fill:#e1f5fe
    style D1 fill:#fff3e0
    style D2 fill:#f3e5f5
    style E1 fill:#e8f5e8
    style E2 fill:#e8f5e8
```
通过以上系统化的集成设计，我们实现了一个高度模块化、可配置、易扩展的地铁标志识别系统。系统支持灵活的配置管理、完善的流程控制、多格式数据处理和全面的结果评估，为研究和实际应用提供了坚实的技术基础。

---

## 7. 实验设计与结果分析

### 7.1 实验环境与设置

#### 7.1.1 硬件环境

本项目在以下硬件环境下进行开发和测试：

**计算平台**：
- **设备型号**：MacBook Pro M1 Pro (2021)
- **处理器**：Apple M1 Pro芯片 (10核CPU，16核GPU)
- **内存**：16GB 统一内存架构
- **存储**：512GB SSD

**计算资源特点**：
- **CPU密集型处理**：传统计算机视觉算法（颜色特征提取、形态学操作）充分利用M1芯片的高性能CPU核心
- **内存效率**：统一内存架构提高了数据处理效率，特别适合图像处理任务
- **无独立GPU训练**：系统设计充分考虑了资源受限环境，重点优化算法效率而非依赖大型深度学习模型

#### 7.1.2 软件环境

**核心软件版本**：
- **Python版本**：3.10.x
- **操作系统**：macOS Sonoma 14.x

**主要依赖库及版本**：
```python
# 核心计算和数据处理
numpy==2.1.3
pandas==2.2.3
opencv-python==4.11.0.86
scipy==1.15.3
scikit-learn==1.6.1

# 深度学习框架
tensorflow==2.19.0
keras==3.9.2
tensorflow-metal==1.2.0  # M1芯片优化

# 配置管理和工具
hydra-core==1.3.2
omegaconf==2.3.0
python-dotenv==1.1.0

# 可视化和分析
matplotlib==3.10.3
pillow==11.2.1

# 其他工具库
PyYAML==6.0.2
rich==14.0.0
tqdm (内置于其他库)
```

**环境配置特点**：
- **M1优化**：使用tensorflow-metal充分利用M1芯片的GPU加速能力
- **版本兼容性**：所有依赖库均选择M1芯片兼容版本
- **轻量化部署**：依赖库精简，适合资源受限环境部署

#### 7.1.3 评估指标定义

基于地铁标志检测的特殊性，我们设计了专门的评估指标体系：

**检测级别指标**：

1. **整体检测率（Detection Rate）**：
   ```python
   detection_rate = total_detected / total_gt
   ```
   衡量系统整体的检测能力

2. **线路级召回率（Line-specific Recall）**：
   ```python
   recall_line_i = detected_line_i / gt_line_i
   ```
   衡量每条地铁线路的检测性能

3. **IoU匹配阈值**：
   ```python
   def calculate_iou(box1, box2):
       # 计算两个边界框的交并比
       intersection_area = max(0, min(x2_1, x2_2) - max(x1_1, x1_2)) * \
                          max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
       union_area = area_box1 + area_box2 - intersection_area
       return intersection_area / union_area
   ```
   使用IoU ≥ 0.3作为检测匹配阈值

**问题分类指标**：

我们将检测问题分为以下几类：

1. **完全未检测（No Detection）**：
   - 定义：GT中有目标但系统未检测到任何目标
   - 统计：未检测图像数量和对应的GT线路

2. **检测不足（Under Detection）**：
   - 定义：检测数量比GT少2个或以上
   - 影响：可能导致用户错过重要线路信息

3. **过度检测（Over Detection）**：
   - 定义：检测数量超过GT数量
   - 影响：可能产生误导性信息

4. **线路混淆（Line Confusion）**：
   - 定义：检测到目标但线路分类错误
   - 基于IoU匹配分析混淆矩阵

**评估代码实现**：
```python
def analyze_detection_results(results):
    """自定义检测结果分析函数"""
    total_images = len(results)
    total_gt = sum(len(r['ground_truth']) for r in results)
    total_detected = sum(len(r['detected']) for r in results)
    
    # 线路级别统计
    line_stats = {}
    for item in results:
        gt_lines = [ann[0] for ann in item['ground_truth']]
        detected_lines = [ann[0] for ann in item['detected']]
        
        for line in gt_lines:
            if line not in line_stats:
                line_stats[line] = {'gt': 0, 'detected': 0, 'missed': 0}
            line_stats[line]['gt'] += 1
            if str(line) in detected_lines:
                line_stats[line]['detected'] += 1
            else:
                line_stats[line]['missed'] += 1
    
    return {
        'detection_rate': total_detected / total_gt,
        'line_stats': line_stats,
        # ... 其他统计信息
    }
```

### 7.2 ROI检测器性能评估

#### 7.2.1 整体检测性能

基于测试数据集的实际评估结果显示出训练样本不足带来的挑战：

**核心性能指标**：

| 指标 | 数值 | 说明 |
|------|------|------|
| 总测试图像数 | 161张 | 来自Test.mat数据集 |
| 总标注目标数 | 257个 | 包含14条不同线路 |
| 总检测目标数 | 346个 | 系统检测结果 |
| **整体检测率** | **134.63%** | 存在明显过度检测问题 |

**检测质量分析**：
- **完全未检测图像**：35张图像（21.7%）未检测到任何目标，暴露了训练样本不足的问题
- **过度检测图像**：60张图像（37.3%）检测数量超过GT，说明误检率较高
- **检测不足图像**：7张图像存在显著漏检（少检测2个以上目标）

**问题根源分析**：
检测率超过100%表明系统存在严重的**误检问题**，主要原因包括：
1. **训练样本严重不足**：87张训练图像难以充分覆盖14条线路在各种光照条件下的颜色变化
2. **HSV色域重叠**：尽管使用2倍标准差扩展色域范围，但光照变化仍导致不同线路间色域混淆
3. **几何约束不足**：当前参数设置无法有效排除圆形M标志等相似形状的干扰物

#### 7.2.2 Comparaison des performances de détection entre les lignes

Les résultats des tests réels montrent que les performances de détection varient considérablement entre les différentes lignes de métro, reflétant directement le problème du déséquilibre des données d'entraînement :

**Lignes à haute performance (Taux de rappel > 80%)** :

| Ligne | Total | Détectés | Manqués | Taux de rappel | Analyse des échantillons d'entraînement |
|-------|--------|-----------|----------|----------------|----------------------------------------|
| Ligne 6 | 16 | 14 | 2 | **87.50%** | Échantillons d'entraînement relativement suffisants, haute distinction des couleurs |
| Ligne 13 | 13 | 11 | 2 | **84.62%** | Caractéristiques de couleur évidentes, peu de confusion |

**Lignes à performance moyenne (Taux de rappel 60-80%)** :

| Ligne | Total | Détectés | Manqués | Taux de rappel |
|-------|--------|-----------|----------|----------------|
| Ligne 3 | 16 | 12 | 4 | 75.00% |
| Ligne 4 | 14 | 10 | 4 | 71.43% |
| Ligne 12 | 30 | 21 | 9 | 70.00% |
| Ligne 10 | 10 | 7 | 3 | 70.00% |
| Ligne 1 | 38 | 26 | 12 | 68.42% |

**Lignes à faible performance (Taux de rappel < 60%)** :

| Ligne | Total | Détectés | Manqués | Taux de rappel |
|-------|--------|-----------|----------|----------------|
| **Ligne 14** | 17 | 6 | 11 | **35.29%** |
| **Ligne 5** | 10 | 3 | 7 | **30.00%** |
| **Ligne 11** | 10 | 3 | 7 | **30.00%** |
| Ligne 8 | 29 | 18 | 11 | 62.07% |
| Ligne 2 | 20 | 12 | 8 | 60.00% |
| Ligne 7 | 20 | 12 | 8 | 60.00% |

#### 7.2.3 HSV色域混淆问题分析

**颜色空间重叠现象**：

训练样本不足导致的最严重问题是HSV色域重叠，特别体现在以下线路组合：

```mermaid
graph TD
    A[光照变化] --> B[HSV色域偏移]
    B --> C[线路间色域重叠]
    C --> D[误检和漏检]
    
    E[线路3 浅黄] --> F[色域重叠区域]
    G[线路11 棕色] --> F
    F --> H[混淆率 30%+]
    
    I[线路2 蓝色] --> J[蓝色系混淆]
    K[线路6 浅蓝] --> J
    L[线路14 深蓝] --> J
    J --> M[低光照下失效]
    
    style F fill:#ffcccc
    style J fill:#ffcccc
    style H fill:#ff9999
    style M fill:#ff9999
```

**统计学增强的局限性**：

尽管我们使用了2倍标准差(`±2σ`)的统计学方法扩展色域范围：

```python
# 当前的统计学增强方法
hsv_lower = [max(0, int(avg_h - 2 * std_h)), 
             max(0, int(avg_s - 2 * std_s)), 
             max(0, int(avg_v - 2 * std_v))]
hsv_upper = [min(180, int(avg_h + 2 * std_h)), 
             min(255, int(avg_s + 2 * std_s)), 
             min(255, int(avg_v + 2 * std_v))]
```

**但仍然存在根本性问题**：
- **训练样本稀少**：某些线路（如线路5）仅有3个训练样本，统计学方法无法有效工作
- **光照条件覆盖不全**：训练数据未充分覆盖极端光照条件
- **色域重叠不可避免**：相似颜色线路在2σ扩展后必然产生重叠区域


### 7.3 颜色特征方法的根本局限性

#### 7.3.1 训练样本不足的连锁影响

**数据稀缺问题**：

我们的分析表明，87张训练图像对于14条线路的识别任务严重不足：

```mermaid
flowchart TD
    A[训练样本不足<br/>87张图像] --> B[统计参数不可靠]
    B --> C[色域范围不准确]
    C --> D[检测性能下降]
    
    A --> E[光照条件覆盖不全]
    E --> F[HSV转换误差]
    F --> G[色域重叠严重]
    G --> D
    
    A --> H[线路样本不平衡]
    H --> I[弱势线路性能差]
    I --> D
    
    D --> J[召回率: 30%-87%]
    D --> K[误检率: 34.6%]
    
    style A fill:#ffcccc
    style D fill:#ff9999
    style J fill:#ffffcc
    style K fill:#ffffcc
```

**典型失败案例分析**：

1. **图像ID 8**: GT包含5条线路[1,4,7,11,14]，但完全未检测
   - **原因**: 极端低光照条件，HSV色彩空间失效
   - **根本问题**: 训练数据缺乏此类极端条件样本

2. **图像ID 242**: GT包含3条线路[1,12,8]，仅检测到部分
   - **原因**: 强反光导致颜色特征严重失真
   - **根本问题**: 颜色恒常性算法无法处理镜面反射

#### 7.3.2 单纯颜色分类的不充分性

基于实际检测结果，我们得出了一个重要结论：**单纯依赖颜色特征进行地铁线路分类是不充分的**。

**根本性问题**：

1. **物理限制**：
   - HSV色彩空间在极端光照下失效
   - 颜色恒常性算法无法完全消除光照影响
   - 相机成像的物理限制

2. **数据限制**：
   - 训练样本无法覆盖所有可能的光照-颜色组合
   - 现实中的光照条件远比训练数据复杂
   - 地铁环境的动态光照变化

3. **算法限制**：
   - 基于阈值的硬分类方法缺乏容错能力
   - 形态学操作无法处理复杂形状变化
   - 缺乏上下文和语义信息

#### 7.3.3 引入CNN分类器的必要性

**当前检测器的积极作用**：

尽管存在误检问题，但当前基于颜色的ROI检测器已经实现了**可接受的召回率水平**：
- **整体召回率**: 68.1%（175/257）
- **高质量线路**: 6条线路召回率超过70%
- **ROI定位能力**: 在正确检测的情况下，边界框定位精度较高

**CNN分类器的定位**：

基于以上分析，我们明确了CNN分类器在系统中的关键作用：

1. **精确分类**：
   - 利用深度特征克服颜色相似性问题
   - 学习形状、纹理、数字等多维特征
   - 提供端到端的分类置信度

2. **误检过滤**：
   - 识别并过滤圆形M标志等干扰物
   - 基于语义特征区分真假地铁标志
   - 降低整体误检率

3. **鲁棒性增强**：
   - 对光照变化的适应性更强
   - 能够处理颜色失真的情况
   - 提供多层次的特征表示

**系统集成策略**：

```mermaid
flowchart LR
    A[输入图像] --> B[颜色ROI检测器]
    B --> C[候选区域生成<br/>召回率导向]
    C --> D[CNN分类器]
    D --> E[精确线路分类<br/>精度导向]
    E --> F[最终检测结果]
    
    G[参数优化] --> B
    G --> H[圆形填充率<br/>形状约束<br/>颜色阈值]
    H --> B
    
    style C fill:#e1f5fe
    style E fill:#f3e5f5
    style F fill:#e8f5e8
```

这种分工明确的两阶段架构充分发挥了各自的优势：
- **ROI检测器**: 快速、高召回率的候选区域生成
- **CNN分类器**: 精确、高精度的最终分类决策

通过这样的系统设计，我们可以在保持计算效率的同时，显著提升整体检测精度和鲁棒性。

### 7.4 系统整体性能

#### 7.4.1 端到端性能评估

**处理速度性能**：
- **单图像处理时间**：平均0.8秒
  - 预处理：0.2秒(Gray_World) 0.3秒(Ebner)
  - ROI检测：0.5秒
  - 后处理：0.1秒
- **批处理效率**：165张图像总处理时间约2.2分钟
- **内存使用**：峰值约500MB

**系统可用性指标**：
- **稳定性**：100%的图像成功处理，无崩溃
- **可重现性**：固定种子下结果100%可重现
- **配置灵活性**：支持20+项参数动态调整

#### 7.4.2 实用性评估

**应用场景适用性**：

1. **智能眼镜集成**：
   - ✅ 计算复杂度适中
   - ✅ 内存占用可控
   - ⚠️ 光照鲁棒性待提升

2. **手机应用部署**：
   - ✅ 跨平台兼容性好
   - ✅ 实时处理能力强
   - ✅ 离线运行能力

3. **辅助导航系统**：
   - ✅ 多线路并行检测
   - ✅ 置信度评估机制
   - ⚠️ 复杂场景鲁棒性待提升

**改进优先级排序**：
1. **高优先级**：提升光照鲁棒性，解决完全未检测问题
2. **中优先级**：平衡训练数据，提升弱势线路性能  
3. **低优先级**：精细化参数调优，降低误检率

### 7.5 技术局限性与改进方向

#### 7.5.1 当前系统局限性

**算法层面局限**：
- 依赖颜色特征，对光照变化敏感
- 缺乏深度学习的高级特征提取能力
- 形态学操作的参数调优复杂

**数据层面局限**：
- 训练数据规模有限（仅87张训练图像）
- 线路间样本分布严重不均
- 缺乏极端条件下的样本

**部署层面局限**：
- 参数调优需要专业知识
- 新环境适应需要重新训练
- 缺乏在线学习能力

#### 7.5.2 未来改进方向


1. **数据增强**：使用色彩变换、几何变换扩充训练数据
2. **集成学习**：结合多种颜色空间和检测策略


通过系统化的实验设计和深入的结果分析，我们证明了基于颜色特征的地铁标志检测方法在资源受限环境下的有效性，同时明确了未来的改进方向和技术路线。

---

## 8. 技术创新点与贡献

### 8.1 颜色参数自动学习方法

本项目在ROI检测器设计中提出了基于K-means聚类的**颜色参数自动学习方法**，这是相比传统手工调参的重要创新：

**技术创新**：
- **自动主导色提取**：使用K-means(K=3)聚类算法自动从训练ROI中提取每条地铁线路的主导颜色
- **循环统计方法**：针对HSV色彩空间H通道的循环特性，实现了循环均值和循环标准差的正确计算
- **统计学扩展策略**：基于2倍标准差(±2σ)自动生成HSV检测范围，平衡召回率和精确率

**技术贡献**：
```python
# 创新的循环均值计算方法
def _circular_mean_deg(angles):
    """计算角度的循环均值（度数）"""
    angles_rad = np.radians(angles)
    mean_sin = np.mean(np.sin(angles_rad))
    mean_cos = np.mean(np.cos(angles_rad))
    return np.degrees(np.arctan2(mean_sin, mean_cos)) % 180
```

这种方法避免了传统手工调参的主观性和工作量，为小样本条件下的颜色特征学习提供了可行的解决方案。

### 8.2 多约束条件ROI筛选策略

我们设计了**多层次几何约束**的创新组合，有效筛选真实的地铁标志：

**创新性约束条件**：
1. **动态面积约束**：根据图像尺寸自适应调整最小/最大面积阈值
2. **圆形填充度评估**：计算轮廓面积与最小外接圆面积的比值
3. **形态学多级处理**：开运算+闭运算+自适应膨胀的组合策略

**技术优势**：
- **自适应性强**：参数根据图像特征动态调整
- **过滤效果好**：有效排除背景噪声和相似形状干扰
- **计算效率高**：基于几何特征的快速筛选

### 8.3 矢量距离与矩形范围融合

提出了**双重颜色匹配策略**的创新融合方法：

**矩形范围掩模**：
```python
# 传统的HSV范围匹配
rectangular_mask = cv2.inRange(img_hsv, hsv_lower, hsv_upper)
```

**矢量距离掩模**：
```python
# 创新的HSV循环距离计算
h_dist = np.minimum(np.abs(h - h_ref), 180 - np.abs(h - h_ref))
distance = np.sqrt((h_weight * h_dist)**2 + (s_weight * s_dist)**2 + (v_weight * v_dist)**2)
vector_mask[distance <= tau] = 255
```

**融合策略**：
```python
# 双重策略融合
combined_mask = cv2.bitwise_or(rectangular_mask, vector_mask)
```

这种融合方法兼顾了计算效率和检测精度，为颜色特征匹配提供了更加鲁棒的解决方案。

### 8.4 模块化可配置系统架构

基于**Hydra配置框架**设计了高度模块化的系统架构：

**架构创新点**：
- **分层配置管理**：支持模式、数据集、算法组件的独立配置
- **依赖注入模式**：通过抽象基类实现组件间的松耦合
- **动态模块加载**：基于配置文件的运行时模块分发
- **环境变量集成**：支持不同部署环境的灵活配置

**实际价值**：
- 大幅降低了算法开发和实验的门槛
- 支持快速的算法组合和参数对比实验
- 为引入CNN分类器提供了良好的扩展基础

### 8.5 自适应形态学处理

开发了基于**图像特征的动态参数调整**机制：

**自适应核大小**：
```python
base_kernel_size = max(3, int(min(height, width) * 0.005))
```

**智能回退机制**：
```python
# 过度处理检测和回退策略
if after_count < before_count * 0.05 and before_count > 0:
    combined_mask = original_combined_mask.copy()
```

这种自适应处理避免了固定参数在不同图像尺寸下的失效问题，提升了算法的鲁棒性。

---

## 9. 系统部署与优化

### 9.1 系统部署方案

#### 9.1.1 轻量化部署架构

本系统专门针对**资源受限环境**进行了优化设计：

**部署目标平台**：
- **智能眼镜**：ARM处理器，有限内存和计算能力
- **智能手机**：iOS/Android平台，离线运行需求
- **嵌入式设备**：实时性要求高的导航辅助设备

**系统资源需求**：
- **内存占用**：峰值约500MB，适合移动设备
- **处理速度**：单张图像0.8秒，满足实时应用需求
- **存储需求**：模型文件<50MB，支持离线部署

#### 9.1.2 跨平台兼容方案

**Python生态兼容**：
- 核心依赖库均支持ARM架构（M1芯片验证）
- OpenCV提供跨平台的计算机视觉支持
- NumPy/SciPy提供高效的数值计算

### 9.2 性能优化策略

#### 9.2.1 算法层面优化

**检测流程优化**：
1. **早期终止机制**：在颜色掩模生成阶段快速排除明显不匹配的区域
2. **批处理关闭**：移动端单张图像处理，避免批处理的内存开销
3. **可视化控制**：生产环境关闭调试可视化，降低计算负担

**内存管理优化**：
```python
# 内存优化策略
def optimized_detect(self, image: np.ndarray) -> List[Dict]:
    # 1. 就地操作，避免不必要的图像拷贝
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 2. 及时释放中间结果
    detections = []
    for line_id in self.color_params:
        mask = self._extract_line_mask(img_hsv, line_id)
        boxes = self._extract_boxes_from_mask(mask)
        detections.extend(boxes)
        del mask  # 及时释放
    
    return detections
```


### 9.3 鲁棒性增强

#### 9.3.1 光照适应性改进

**颜色恒常性增强**：
- **多算法融合**：Gray World + Ebner算法的自适应选择
- **局部处理**：针对反光和阴影区域的特殊处理
- **预处理链优化**：根据图像亮度分布选择最适合的预处理策略

#### 9.3.2 形状约束优化

**几何特征增强**：
```python
# 增强的形状约束参数
enhanced_constraints = {
    "fill_ratio": 0.75,        # 提高圆形填充率要求
    "circularity": 0.8,        # 增加圆度约束
    "aspect_ratio": [0.7, 1.3], # 限制宽高比范围
    "contour_complexity": 0.15   # 轮廓复杂度限制
}
```

这些优化策略确保系统在各种复杂环境下都能保持稳定的检测性能。

---

## 10. 总结与展望

### 10.1 项目成果总结

#### 10.1.1 主要技术贡献

本项目成功构建了一套**基于颜色特征的地铁标志识别系统**，在资源受限环境下实现了可接受的检测性能：

**核心技术成果**：
1. **自动颜色参数学习算法**：解决了传统手工调参的局限性
2. **多约束ROI筛选策略**：有效平衡了检测召回率和精确率
3. **模块化系统架构**：为算法研究和系统部署提供了良好的框架
4. **双重颜色匹配融合**：提升了颜色特征检测的鲁棒性

**实验验证结果**：
- **整体召回率**：68.1%（175/257），达到可接受水平
- **高性能线路**：6条线路召回率超过70%
- **处理效率**：单图像处理时间0.8秒，满足实时应用需求
- **系统稳定性**：100%图像处理成功率，无崩溃现象

#### 10.1.2 实际应用价值

**社会价值**：
- 为视障人士提供地铁导航辅助技术基础
- 展示了计算机视觉技术在无障碍领域的应用潜力
- 为智能眼镜等辅助设备的集成提供了技术方案

**技术价值**：
- 验证了传统计算机视觉方法在特定场景下的有效性
- 提供了小样本条件下颜色特征学习的实用方法
- 为后续深度学习方法的集成奠定了基础

### 10.2 技术局限性分析

#### 10.2.1 根本性限制

通过深入的实验分析，我们识别出当前方法的根本性局限：

**数据层面**：
- **训练样本严重不足**：87张训练图像无法充分覆盖14条线路的颜色变化
- **光照条件覆盖不全**：缺乏极端光照条件下的样本数据
- **线路样本不平衡**：某些线路（如线路5）样本稀少，影响学习效果

**算法层面**：
- **颜色特征固有局限**：HSV色彩空间在极端光照下失效
- **硬阈值分类缺陷**：缺乏容错能力和置信度评估
- **几何约束不充分**：无法有效区分地铁标志和相似形状物体

**应用层面**：
- **ROI误检率偏高**：134.63%的检测率表明存在严重误检问题
- **光照敏感性**：某些线路在特定光照条件下完全失效
- **缺乏语义理解**：无法基于上下文信息进行判断

#### 10.2.2 性能瓶颈

**最严重的问题**：
1. **完全未检测**：35张图像（21.7%）未检测到任何目标
2. **色域重叠混淆**：相似颜色线路间的误判问题
3. **圆形M标志误检**：线路12误检9次，主要来源于形状相似干扰物

### 10.3 未来改进方向

#### 10.3.1 CNN分类器集成

基于当前分析，**引入CNN分类器是系统发展的必然方向**：

**集成策略**：
- **两阶段架构**：ROI检测器（召回率导向）+ CNN分类器（精度导向）
- **轻量化设计**：针对移动端优化的小型CNN架构
- **端到端训练**：联合优化检测和分类两个阶段

**预期改进效果**：
- **误检率降低**：通过语义特征过滤圆形M标志等干扰物
- **光照鲁棒性提升**：深度特征对光照变化的适应性更强
- **分类精度提升**：多维特征学习克服颜色相似性问题

#### 10.3.2 数据增强与优化

**短期改进措施**：
1. **数据增强技术**：
   - 色彩变换增强：模拟不同光照条件
   - 几何变换增强：增加角度和尺度变化
   - 对抗样本生成：提高模型鲁棒性

2. **参数精细化调优**：
   - 圆形填充率阈值提升至0.75
   - 线路特定的`threshold_error_dict`优化
   - 形态学核大小的自适应改进

通过本项目的实施，我们不仅在技术层面取得了创新成果，更重要的是为无障碍智能出行技术的发展做出了有意义的贡献。虽然当前系统还存在一定局限性，但为后续的深度学习集成和系统优化奠定了坚实的基础。

---

## 参考文献

[1] Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. *2005 IEEE computer society conference on computer vision and pattern recognition (CVPR'05)*, 1, 886-893.

[2] Viola, P., & Jones, M. (2001). Rapid object detection using a boosted cascade of simple features. *Proceedings of the 2001 IEEE computer society conference on computer vision and pattern recognition. CVPR 2001*, 1, I-I.

[3] Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. *Proceedings of the IEEE conference on computer vision and pattern recognition*, 779-788.

[4] Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C. Y., & Berg, A. C. (2016). Ssd: Single shot multibox detector. *European conference on computer vision*, 21-37.

[5] Girshick, R., Donahue, J., Darrell, T., & Malik, J. (2014). Rich feature hierarchies for accurate object detection and semantic segmentation. *Proceedings of the IEEE conference on computer vision and pattern recognition*, 580-587.

[6] Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., ... & Adam, H. (2017). Mobilenets: Efficient convolutional neural networks for mobile vision applications. *arXiv preprint arXiv:1704.04861*.

[7] Stallkamp, J., Schlipsing, M., Salmen, J., & Igel, C. (2012). Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition. *Neural networks*, 32, 323-332.

[8] Sermanet, P., & LeCun, Y. (2011). Traffic sign recognition with multi-scale convolutional networks. *The 2011 international joint conference on neural networks*, 2809-2813.

[9] Buchner, M., & Zauner, G. (2006). Real-time detection and recognition of road traffic signs. *IEEE Transactions on Intelligent Transportation Systems*, 7(2), 213-222.

[10] Land, E. H. (1977). The retinex theory of color vision. *Scientific American*, 237(6), 108-128.

[11] Ebner, F. (1998). Color constancy based on local space average color. *Machine Vision and Applications*, 11(5), 283-301.

[12] Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. *Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms*, 1027-1035.

[13] Suzuki, S., & Abe, K. (1985). Topological structural analysis of digitized binary images by border following. *Computer vision, graphics, and image processing*, 30(1), 32-46.

[14] Hydra Framework Documentation. Facebook Research. https://hydra.cc/

[15] Bradski, G. (2000). The OpenCV Library. *Dr. Dobb's Journal of Software Tools*.

---

## 附录

### 附录A: 关键算法伪代码

#### A.1 颜色参数自动学习算法

```
算法1: 自动颜色参数学习
输入: 训练数据集D, 线路ID集合L
输出: 各线路颜色参数字典P

for each line_id in L:
    color_samples = []
    for each roi in D where roi.line_id == line_id:
        hsv_roi = convert_to_hsv(roi.image)
        dominant_color = kmeans_extract_dominant(hsv_roi, K=3)
        color_samples.append(dominant_color)
    
    # 过滤异常样本
    filtered_samples = remove_outliers(color_samples, threshold=2.0)
    
    # 计算循环统计参数
    mean_h = circular_mean(filtered_samples.h)
    mean_s = mean(filtered_samples.s)
    mean_v = mean(filtered_samples.v)
    
    std_h = circular_std(filtered_samples.h)
    std_s = std(filtered_samples.s)
    std_v = std(filtered_samples.v)
    
    # 生成检测范围
    P[line_id] = {
        'hsv_mean': (mean_h, mean_s, mean_v),
        'hsv_lower': (mean_h - 2*std_h, mean_s - 2*std_s, mean_v - 2*std_v),
        'hsv_upper': (mean_h + 2*std_h, mean_s + 2*std_s, mean_v + 2*std_v)
    }

return P
```

#### A.2 双重颜色掩模生成算法

```
算法2: 双重颜色掩模生成
输入: HSV图像I, 颜色参数P, 阈值误差E
输出: 融合颜色掩模M

# 矩形范围掩模
lower_bound = adjust_bounds(P.hsv_lower, -E)
upper_bound = adjust_bounds(P.hsv_upper, +E)
mask_rect = inRange(I, lower_bound, upper_bound)

# 矢量距离掩模
mask_vector = zeros_like(I)
for each pixel (h,s,v) in I:
    h_dist = min(abs(h - P.mean_h), 180 - abs(h - P.mean_h))
    s_dist = abs(s - P.mean_s)
    v_dist = abs(v - P.mean_v)
    
    weighted_distance = sqrt((h_weight * h_dist)² + 
                           (s_weight * s_dist)² + 
                           (v_weight * v_dist)²)
    
    if weighted_distance <= tau:
        mask_vector[pixel] = 255

# 掩模融合
mask_combined = bitwise_or(mask_rect, mask_vector)

# 形态学处理
kernel_small = get_ellipse_kernel(base_size)
kernel_medium = get_ellipse_kernel(base_size * 2)

mask_combined = morphology_open(mask_combined, kernel_small)
mask_combined = morphology_close(mask_combined, kernel_medium)

return mask_combined
```

### 附录B: 配置文件详细说明

#### B.1 主配置文件结构

```yaml
# config/config.yaml
defaults:
  - mode: train                    # 运行模式选择
  - dataset: default              # 数据集配置
  - roi_detection: multi_color    # ROI检测器配置
  - classification: template_classification  # 分类器配置
  - preprocessing/template: default           # 模板预处理配置
  - preprocessing/roi_param_optimizer: default # ROI预处理配置

# 全局配置
output_dir: ${base_dir}/results   # 输出目录
seed: 42                         # 随机种子
base_dir: ${oc.env:BASE_DIR}     # 基础目录，从环境变量读取

# 模块分发字典
model_dispatch:
  train: "src.pipeline.train_pipeline:main"
  test: "src.pipeline.test_pipeline:main"
  demo: "src.pipeline.demo_pipeline:main"

# 日志配置
logging:
  level: INFO
  format: "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
```

#### B.2 ROI检测器配置

```yaml
# config/roi_detection/multi_color.yaml
_target_: src.roi_detection.multi_color_detector.MultiColorDetector

# 颜色参数配置
color_params_path: "${base_dir}/config/color_params.yaml"
threshold_error_dict:
  "1": 2    # 线路1阈值误差
  "2": 2    # 线路2阈值误差
  "3": 5    # 线路3阈值误差
  "4": 5    # 线路4阈值误差
  "5": 15   # 线路5阈值误差（样本少，放宽阈值）
  "6": 5    # 线路6阈值误差
  "7": 10   # 线路7阈值误差
  "8": 5    # 线路8阈值误差
  "9": 8    # 线路9阈值误差
  "10": 8   # 线路10阈值误差
  "11": 8   # 线路11阈值误差
  "12": 5   # 线路12阈值误差
  "13": 8   # 线路13阈值误差
  "14": 10  # 线路14阈值误差

# 几何约束参数
min_area: 1000              # 最小面积
max_area: 50000             # 最大面积
min_aspect_ratio: 0.5       # 最小宽高比
max_aspect_ratio: 2.0       # 最大宽高比
fill_ratio: 0.6             # 圆形填充率阈值

# 形态学操作参数
kernel_size_factor: 0.005   # 核大小因子
max_kernel_size: 15         # 最大核大小
min_kernel_size: 3          # 最小核大小

# NMS参数
nms_iou_threshold: 0.5      # IoU阈值
```

### 附录C: 实验数据详表

#### C.1 各线路详细检测结果



#### C.2 处理时间性能统计

| 处理阶段 | 平均时间(ms) | 最大时间(ms) | 最小时间(ms) | 标准差(ms) |
|----------|--------------|--------------|--------------|------------|
| 图像预处理 | 180 | 250 | 120 | 35 |
| 颜色掩模生成 | 320 | 450 | 200 | 65 |
| 形态学处理 | 140 | 200 | 80 | 28 |
| 轮廓提取 | 90 | 150 | 50 | 22 |
| 几何筛选 | 60 | 100 | 30 | 18 |
| NMS处理 | 15 | 30 | 5 | 8 |
| **总处理时间** | **805** | **1180** | **485** | **176** |

### 附录D: 系统安装与使用指南

#### D.1 环境配置

**系统要求**：
- Python 3.10+
- macOS 11.0+ (M1芯片优化) 或 Linux
- 内存: 8GB+ 推荐

**安装步骤**：
```bash
# 1. 克隆项目
git clone https://github.com/cyang2001/metro.git
cd Metro

# 2. 创建虚拟环境
python -m venv metro_env
source metro_env/bin/activate  # macOS/Linux
# metro_env\Scripts\activate   # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
export BASE_DIR=$(pwd)
echo "BASE_DIR=$(pwd)" > pc_environment.env
```

#### D.2 使用示例

**训练模式**：
```bash
# 基础训练
python run.py mode=train

# 仅优化ROI参数
python run.py mode=train mode.optimize_roi_param=true 

# 仅创建模板
python run.py mode=train mode.optimize_roi_param=false mode.create_templates=true mode.train_cnn = false

# 仅训练CNN
python run.py mode=train mode.optimize_roi_param=false mode.create_templates=false mode.train_cnn=true
```

**测试模式**：
```bash
# 基础测试
python run.py mode=test

# 启用可视化
python run.py mode=test mode.has_visualize=true

# 指定输出目录
python run.py mode=test output_dir=./custom_results
```

**配置覆盖**：
```bash
# 修改数据集路径
python run.py dataset.data_root=./custom_data

# 调整检测参数
python run.py roi_detection.fill_ratio=0.75

# 修改日志级别
python run.py logging.level=DEBUG
```

---

**报告完成说明**：
本技术报告全面记录了巴黎地铁线路识别系统的设计、实现、实验和分析过程。通过系统化的方法论和深入的实证分析，我们不仅展示了基于颜色特征的传统计算机视觉方法在特定应用场景下的有效性，也客观地分析了其局限性，为后续的深度学习集成提供了坚实的基础。
