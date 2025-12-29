---

# **PP-YOLOE**

PP-YOLOE is an **industrial-grade, state-of-the-art object detector** designed for **high performance** and **easy deployment**. Built on the foundation of **PP-YOLOv2**, PP-YOLOE introduces several advanced features, including an **anchor-free paradigm**, a **powerful backbone and neck** with **CSPRepResStage**, **ET-Head**, and the **dynamic label assignment algorithm TAL (Task Alignment Learning)**.

![PP-YOLOE](https://github.com/Gaurav14cs17/YOLOE/blob/main/images/yoloe.png)

---

## **Key Features**
- **Anchor-Free Paradigm**: Simplifies the detection pipeline and improves performance.
- **CSPRepResStage**: A powerful backbone and neck architecture for enhanced feature extraction.
- **ET-Head**: Efficient and accurate detection head.
- **TAL (Task Alignment Learning)**: Dynamic label assignment algorithm for better training.
- **High Performance**: Achieves state-of-the-art results on benchmark datasets.
- **Easy Deployment**: Optimized for industrial applications with minimal setup.

---

## **Getting Started**

### **1. Installation**
To get started with PP-YOLOE, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Gaurav14cs17/YOLOE.git
   cd YOLOE
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the pretrained weights and configuration files from the [PaddlePaddle Model Zoo](https://github.com/PaddlePaddle/PaddleDetection).

### **2. Inference**
Run inference on an image, video, or webcam using the provided scripts:

```bash
# Run on image
python inference.py \
    --source /path/to/image.jpg \
    --weights /path/to/model.pt \
    --model-size s \
    --conf-thres 0.25

# Run on video
python inference.py \
    --source /path/to/video.mp4 \
    --weights /path/to/model.pt

# Run on webcam (use webcam index)
python inference.py \
    --source 0 \
    --weights /path/to/model.pt

# Run on directory of images
python inference.py \
    --source /path/to/images/ \
    --weights /path/to/model.pt \
    --save-txt
```

### **3. Training**
To train PP-YOLOE on your custom dataset:

#### COCO Format Dataset:
```bash
python train.py \
    --data-dir /path/to/coco \
    --train-ann /path/to/annotations/instances_train2017.json \
    --val-ann /path/to/annotations/instances_val2017.json \
    --model-size s \
    --epochs 300 \
    --batch-size 16 \
    --img-size 640 \
    --lr 0.01 \
    --amp  # Enable mixed precision training
```

#### YOLO Format Dataset:
```bash
python train.py \
    --data-dir /path/to/custom_data \
    --model-size s \
    --num-classes 10 \
    --epochs 100 \
    --batch-size 16
```

#### Training Options:
- `--model-size`: Model size (s, m, l, x)
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size per GPU
- `--img-size`: Input image size
- `--lr`: Initial learning rate
- `--amp`: Enable automatic mixed precision
- `--resume`: Resume from checkpoint

---

## **Model Architecture**

### **Backbone: CSPRepResStage**
The backbone of PP-YOLOE is based on **CSPRepResStage**, which enhances feature extraction capabilities while maintaining efficiency.

### **Neck: ET-Head**
The **ET-Head** is a lightweight and efficient detection head that improves accuracy and reduces computational overhead.

### **Anchor-Free Paradigm**
PP-YOLOE adopts an **anchor-free approach**, which simplifies the detection pipeline and eliminates the need for anchor box tuning.

### **TAL (Task Alignment Learning)**
The **Task Alignment Learning (TAL)** algorithm dynamically assigns labels during training, improving the alignment between classification and localization tasks.

---

## **Performance**
PP-YOLOE achieves **state-of-the-art performance** on benchmark datasets like COCO. Below are some key metrics:

| Model           | mAP (COCO) | FPS  |
|-----------------|------------|------|
| PP-YOLOE-L      | 50.9       | 78   |
| PP-YOLOE-X      | 52.2       | 68   |

---

## **Results**

### **Object Detection Demo**
Check out the object detection results using PP-YOLOE:

![PP-YOLOE Demo](https://github.com/Gaurav14cs17/YOLOE/blob/main/images/yoloe.png)

---

## **References**
- **PP-YOLOE Paper**: [PP-YOLOE: An Evolved Version of YOLO](https://arxiv.org/abs/2108.06753)
- **PaddlePaddle Detection Repository**: [GitHub Repository](https://github.com/PaddlePaddle/PaddleDetection)

---

## **Folder Structure**
Here's an overview of the repository structure:

```
YOLOE/
├── models/                   # Model architecture
│   ├── __init__.py
│   ├── backbone.py           # CSPResNet backbone
│   ├── pan.py                # CustomCSPPAN neck (FPN + PAN)
│   ├── head.py               # PPYOLOEHead detection head
│   ├── yoloe.py              # Main YOLOE model
│   └── iou_loss.py           # IoU loss functions (GIoU, DIoU, CIoU)
├── assigners/                # Target assignment utilities
│   ├── __init__.py
│   ├── atss_assigner.py      # ATSS assigner
│   ├── bbox_utils.py         # Bounding box utilities
│   └── utils.py              # Helper functions
├── images/                   # Demo images and visualizations
│   └── yoloe.png
├── train.py                  # Training script
├── inference.py              # Inference script (image/video/webcam)
├── dataset.py                # Dataset classes (COCO, YOLO format)
├── requirements.txt          # Python dependencies
└── README.md                 # Project README file
```

## **Model Sizes**

| Model | Width Mult | Depth Mult | Parameters |
|-------|------------|------------|------------|
| YOLOE-S | 0.50 | 0.33 | ~7M |
| YOLOE-M | 0.75 | 0.67 | ~25M |
| YOLOE-L | 1.00 | 1.00 | ~50M |
| YOLOE-X | 1.25 | 1.33 | ~100M |

## **Dataset Formats**

### COCO Format
Standard COCO JSON annotation format with `images`, `annotations`, and `categories`.

### YOLO Format
```
data/
├── train/
│   ├── images/
│   │   ├── img001.jpg
│   │   └── img002.jpg
│   └── labels/
│       ├── img001.txt
│       └── img002.txt
└── val/
    ├── images/
    └── labels/
```

Label file format (normalized coordinates):
```
class_id x_center y_center width height
```

---

## **How to Contribute**
We welcome contributions! If you'd like to contribute to this project, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes.
4. Submit a pull request.

---

## **License**
This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.

---

## **Acknowledgments**
- Thanks to the **PaddlePaddle team** for developing and open-sourcing PP-YOLOE.
- Special thanks to the contributors of the **PaddleDetection repository** for their support and tools.
---
