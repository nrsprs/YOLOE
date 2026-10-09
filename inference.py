"""
YOLOE Inference Script
Supports image, video, and webcam inference
"""
import os
import sys
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

from models.yoloe import build_yoloe


# COCO class names
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Colors for visualization
COLORS = np.random.randint(0, 255, size=(len(COCO_CLASSES), 3), dtype=np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOE Inference')
    
    # Input
    parser.add_argument('--source', type=str, required=True,
                        help='Path to image, video, directory, or webcam index (0, 1, ...)')
    
    # Model
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--model-size', type=str, default='s', choices=['s', 'm', 'l', 'x'])
    parser.add_argument('--num-classes', type=int, default=80)
    parser.add_argument('--img-size', type=int, default=640, help='Input image size')
    
    # Inference
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300, help='Maximum detections per image')
    parser.add_argument('--classes', nargs='+', type=int, help='Filter by class')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./runs/detect', help='Output directory')
    parser.add_argument('--save-txt', action='store_true', help='Save results to txt files')
    parser.add_argument('--save-img', action='store_true', default=True, help='Save result images')
    parser.add_argument('--show', action='store_true', help='Display results')
    
    # System
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    return parser.parse_args()


class YOLOEPredictor:
    """
    YOLOE predictor for inference
    """
    
    def __init__(self, weights, model_size='s', num_classes=80, img_size=640,
                 conf_thres=0.25, iou_thres=0.45, max_det=300, device='cuda',
                 class_names=None):
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names or COCO_CLASSES
        
        # Build and load model
        print(f"Loading YOLOE-{model_size} model...")
        self.model = build_yoloe(model_size, num_classes)
        
        # Load weights
        state_dict = torch.load(weights, map_location=self.device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        self.model.load_state_dict(state_dict)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
        
        # Warmup
        self._warmup()
    
    def _warmup(self):
        """Warmup the model"""
        dummy = torch.zeros(1, 3, self.img_size, self.img_size, device=self.device)
        for _ in range(3):
            with torch.no_grad():
                self.model(dummy)
        print("Model warmed up")
    
    def preprocess(self, image):
        """
        Preprocess image for inference
        
        Args:
            image: numpy array (H, W, C) in BGR format or PIL Image
            
        Returns:
            tensor: preprocessed image tensor
            orig_size: original image size (H, W)
            scale: scale factor used for resizing
        """
        if isinstance(image, np.ndarray):
            # BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        orig_w, orig_h = image.size
        
        # PP-YOLOE eval resizes to the input size directly (no letterbox)
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        
        # To tensor (to_tensor already scales pixels to [0, 1], PP-YOLOE uses no mean/std)
        tensor = TF.to_tensor(image)
        tensor = tensor.unsqueeze(0)
        
        scale = (self.img_size / orig_w, self.img_size / orig_h)
        return tensor.to(self.device), (orig_h, orig_w), scale
    
    def postprocess(self, predictions, orig_size, scale, filter_classes=None):
        """
        Post-process model predictions
        
        Args:
            predictions: (cls_scores, reg_dists) from model
            orig_size: original image size (H, W)
            scale: scale factor used in preprocessing
            filter_classes: list of class indices to keep
            
        Returns:
            List of detections, each as (x1, y1, x2, y2, conf, class_id)
        """
        cls_scores, reg_dists = predictions
        
        # Generate anchor points
        h, w = self.img_size, self.img_size
        strides = [32, 16, 8]
        feat_sizes = [(h // s, w // s) for s in strides]
        
        anchor_points = []
        stride_tensor = []
        
        for feat_size, stride in zip(feat_sizes, strides):
            fh, fw = feat_size
            shift_x = (torch.arange(fw, device=self.device, dtype=torch.float32) + 0.5) * stride
            shift_y = (torch.arange(fh, device=self.device, dtype=torch.float32) + 0.5) * stride
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
            anchor_point = torch.stack([shift_x, shift_y], dim=-1).reshape(-1, 2)
            anchor_points.append(anchor_point)
            stride_tensor.append(torch.full((fh * fw, 1), stride, device=self.device, dtype=torch.float32))
        
        anchor_points = torch.cat(anchor_points, dim=0)
        stride_tensor = torch.cat(stride_tensor, dim=0)
        
        # Decode boxes from reg_dists
        # reg_dists: (B, 4, total_anchors)
        reg_dists = reg_dists.permute(0, 2, 1)  # (B, total_anchors, 4)
        
        # Convert to xyxy
        x1 = anchor_points[:, 0] - reg_dists[0, :, 0] * stride_tensor.squeeze()
        y1 = anchor_points[:, 1] - reg_dists[0, :, 1] * stride_tensor.squeeze()
        x2 = anchor_points[:, 0] + reg_dists[0, :, 2] * stride_tensor.squeeze()
        y2 = anchor_points[:, 1] + reg_dists[0, :, 3] * stride_tensor.squeeze()
        
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # (total_anchors, 4)
        
        # Get class scores and predictions
        cls_scores = cls_scores.permute(0, 2, 1)[0]  # (total_anchors, num_classes)
        
        # Get max class score and class id
        max_scores, class_ids = cls_scores.max(dim=-1)
        
        # Filter by confidence
        keep = max_scores > self.conf_thres
        
        if not keep.any():
            return []
        
        boxes = boxes[keep]
        scores = max_scores[keep]
        class_ids = class_ids[keep]
        
        # Filter by class if specified
        if filter_classes is not None:
            class_mask = torch.zeros_like(class_ids, dtype=torch.bool)
            for cls_id in filter_classes:
                class_mask |= (class_ids == cls_id)
            boxes = boxes[class_mask]
            scores = scores[class_mask]
            class_ids = class_ids[class_mask]
        
        if len(boxes) == 0:
            return []
        
        # NMS
        from torchvision.ops import nms
        keep_indices = nms(boxes, scores, self.iou_thres)
        keep_indices = keep_indices[:self.max_det]
        
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        class_ids = class_ids[keep_indices]
        
        # Scale boxes back to original image size (per-axis, input is not letterboxed)
        scale_x, scale_y = scale
        boxes[:, [0, 2]] = boxes[:, [0, 2]] / scale_x
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / scale_y
        
        # Clip to image boundaries
        orig_h, orig_w = orig_size
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, orig_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, orig_h)
        
        # Convert to list
        detections = []
        for box, score, cls_id in zip(boxes, scores, class_ids):
            detections.append({
                'box': box.cpu().numpy(),
                'score': score.item(),
                'class_id': cls_id.item(),
                'class_name': self.class_names[cls_id.item()] if cls_id.item() < len(self.class_names) else str(cls_id.item())
            })
        
        return detections
    
    @torch.no_grad()
    def predict(self, image, filter_classes=None):
        """
        Run inference on an image
        
        Args:
            image: numpy array (H, W, C) in BGR format or PIL Image
            filter_classes: list of class indices to keep
            
        Returns:
            List of detections
        """
        # Preprocess
        tensor, orig_size, scale = self.preprocess(image)
        
        # Inference
        predictions = self.model(tensor)
        
        # Postprocess
        detections = self.postprocess(predictions, orig_size, scale, filter_classes)
        
        return detections


def draw_detections(image, detections, class_names=None):
    """
    Draw detection boxes on image
    
    Args:
        image: numpy array (H, W, C) in BGR format
        detections: list of detection dicts
        class_names: list of class names
        
    Returns:
        image with drawn detections
    """
    image = image.copy()
    
    for det in detections:
        box = det['box'].astype(int)
        score = det['score']
        cls_id = det['class_id']
        cls_name = det['class_name']
        
        # Get color
        color = tuple(map(int, COLORS[cls_id % len(COLORS)]))
        
        # Draw box
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
        
        # Draw label
        label = f"{cls_name}: {score:.2f}"
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        y1 = max(box[1], label_size[1] + 10)
        cv2.rectangle(image, (box[0], y1 - label_size[1] - 10),
                      (box[0] + label_size[0], y1), color, -1)
        cv2.putText(image, label, (box[0], y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image


def process_image(predictor, image_path, output_dir, save_txt=False, save_img=True, show=False, filter_classes=None):
    """Process a single image"""
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Run inference
    start_time = time.time()
    detections = predictor.predict(image, filter_classes)
    inference_time = time.time() - start_time
    
    print(f"{image_path}: {len(detections)} detections in {inference_time*1000:.1f}ms")
    
    # Save results
    output_path = output_dir / Path(image_path).name
    
    if save_txt:
        txt_path = output_path.with_suffix('.txt')
        with open(txt_path, 'w') as f:
            for det in detections:
                box = det['box']
                f.write(f"{det['class_id']} {det['score']:.4f} "
                        f"{box[0]:.1f} {box[1]:.1f} {box[2]:.1f} {box[3]:.1f}\n")
    
    if save_img:
        result_image = draw_detections(image, detections)
        cv2.imwrite(str(output_path), result_image)
    
    if show:
        result_image = draw_detections(image, detections)
        cv2.imshow('YOLOE Detection', result_image)
        cv2.waitKey(0)
    
    return detections


def process_video(predictor, video_path, output_dir, save_img=True, show=False, filter_classes=None):
    """Process a video file"""
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
    
    # Setup video writer
    writer = None
    if save_img:
        output_path = output_dir / Path(video_path).name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_idx = 0
    total_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        start_time = time.time()
        detections = predictor.predict(frame, filter_classes)
        inference_time = time.time() - start_time
        total_time += inference_time
        
        # Draw detections
        result_frame = draw_detections(frame, detections)
        
        # Add FPS info
        fps_text = f"FPS: {1/inference_time:.1f}"
        cv2.putText(result_frame, fps_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if writer:
            writer.write(result_frame)
        
        if show:
            cv2.imshow('YOLOE Detection', result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_idx += 1
        if frame_idx % 100 == 0:
            avg_fps = frame_idx / total_time
            print(f"Processed {frame_idx}/{total_frames} frames, Avg FPS: {avg_fps:.1f}")
    
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    avg_fps = frame_idx / total_time
    print(f"Done! Average FPS: {avg_fps:.1f}")


def process_webcam(predictor, webcam_id, output_dir=None, save_img=False, filter_classes=None):
    """Process webcam stream"""
    cap = cv2.VideoCapture(webcam_id)
    
    if not cap.isOpened():
        print(f"Error: Could not open webcam {webcam_id}")
        return
    
    print("Press 'q' to quit, 's' to save current frame")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        start_time = time.time()
        detections = predictor.predict(frame, filter_classes)
        inference_time = time.time() - start_time
        
        # Draw detections
        result_frame = draw_detections(frame, detections)
        
        # Add FPS info
        fps_text = f"FPS: {1/inference_time:.1f} | Detections: {len(detections)}"
        cv2.putText(result_frame, fps_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('YOLOE Webcam Detection', result_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and output_dir:
            save_path = output_dir / f'webcam_frame_{frame_count}.jpg'
            cv2.imwrite(str(save_path), result_frame)
            print(f"Saved: {save_path}")
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize predictor
    predictor = YOLOEPredictor(
        weights=args.weights,
        model_size=args.model_size,
        num_classes=args.num_classes,
        img_size=args.img_size,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        max_det=args.max_det,
        device=args.device
    )
    
    source = args.source
    
    # Check if webcam
    if source.isdigit():
        process_webcam(predictor, int(source), output_dir, args.save_img, args.classes)
    
    # Check if video
    elif source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        process_video(predictor, source, output_dir, args.save_img, args.show, args.classes)
    
    # Check if directory
    elif os.path.isdir(source):
        image_files = []
        for ext in ['jpg', 'jpeg', 'png', 'bmp', 'webp']:
            image_files.extend(Path(source).glob(f'*.{ext}'))
            image_files.extend(Path(source).glob(f'*.{ext.upper()}'))
        
        print(f"Found {len(image_files)} images")
        
        for img_path in image_files:
            process_image(predictor, img_path, output_dir,
                          args.save_txt, args.save_img, args.show, args.classes)
    
    # Single image
    else:
        process_image(predictor, source, output_dir,
                      args.save_txt, args.save_img, args.show, args.classes)
    
    print(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()

