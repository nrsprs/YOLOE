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
Run inference on an image or video using the provided scripts:
```bash
python infer.py --config configs/ppyoloe_crn_l_300e_coco.yml --model_path weights/ppyoloe_crn_l_300e_coco.pdparams --image_path input_image.jpg
```

### **3. Training**
To train PP-YOLOE on your custom dataset:
1. Prepare your dataset in COCO format.
2. Modify the configuration file (`configs/ppyoloe_crn_l_300e_coco.yml`) to point to your dataset.
3. Start training:
   ```bash
   python tools/train.py -c configs/ppyoloe_crn_l_300e_coco.yml
   ```

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
Here’s an overview of the repository structure:

```
YOLOE/
├── configs/                  # Configuration files for training and inference
│   └── ppyoloe_crn_l_300e_coco.yml
├── weights/                  # Pretrained model weights
│   └── ppyoloe_crn_l_300e_coco.pdparams
├── tools/                    # Training and evaluation scripts
│   └── train.py
├── images/                   # Demo images and visualizations
│   └── yoloe.png
├── infer.py                  # Inference script
└── README.md                 # Project README file
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
