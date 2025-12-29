"""
Dataset classes for YOLOE training
Supports COCO format and custom datasets
"""
import os
import json
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


class COCODataset(Dataset):
    """
    COCO format dataset for object detection
    
    Args:
        img_dir: Path to image directory
        ann_file: Path to COCO format annotation JSON file
        img_size: Target image size (int or tuple)
        augment: Whether to apply data augmentation
        cache_images: Whether to cache images in memory
    """
    
    def __init__(self, img_dir, ann_file, img_size=640, augment=True, cache_images=False):
        self.img_dir = img_dir
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.augment = augment
        self.cache_images = cache_images
        
        # Load annotations
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
        
        # Build image id to filename mapping
        self.images = {img['id']: img for img in coco_data['images']}
        
        # Build category id to index mapping
        self.cat_ids = sorted([cat['id'] for cat in coco_data['categories']])
        self.cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(self.cat_ids)}
        self.num_classes = len(self.cat_ids)
        
        # Group annotations by image
        self.img_ids = list(self.images.keys())
        self.annotations = {img_id: [] for img_id in self.img_ids}
        
        for ann in coco_data['annotations']:
            if ann['image_id'] in self.annotations:
                self.annotations[ann['image_id']].append(ann)
        
        # Filter out images without annotations
        self.img_ids = [img_id for img_id in self.img_ids if len(self.annotations[img_id]) > 0]
        
        # Image cache
        self.cached_images = {}
        if cache_images:
            print("Caching images...")
            for img_id in self.img_ids:
                self.cached_images[img_id] = self._load_image(img_id)
        
        print(f"Loaded {len(self.img_ids)} images with annotations")
    
    def __len__(self):
        return len(self.img_ids)
    
    def _load_image(self, img_id):
        """Load image from disk"""
        img_info = self.images[img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        return img
    
    def _get_annotations(self, img_id):
        """Get bounding boxes and labels for an image"""
        anns = self.annotations[img_id]
        
        boxes = []
        labels = []
        
        for ann in anns:
            if ann.get('iscrowd', 0):
                continue
            
            # COCO format: [x, y, width, height]
            x, y, w, h = ann['bbox']
            
            # Convert to [x1, y1, x2, y2]
            x1, y1, x2, y2 = x, y, x + w, y + h
            
            if w > 0 and h > 0:
                boxes.append([x1, y1, x2, y2])
                labels.append(self.cat_id_to_idx[ann['category_id']])
        
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        
        return boxes, labels
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        # Load image
        if self.cache_images and img_id in self.cached_images:
            img = self.cached_images[img_id].copy()
        else:
            img = self._load_image(img_id)
        
        # Get annotations
        boxes, labels = self._get_annotations(img_id)
        
        # Get original size
        orig_w, orig_h = img.size
        
        # Apply augmentations
        if self.augment:
            img, boxes = self._augment(img, boxes)
        
        # Resize image
        img, boxes = self._resize(img, boxes)
        
        # Convert to tensor
        img = TF.to_tensor(img)
        
        # Normalize
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'orig_size': torch.tensor([orig_h, orig_w])
        }
        
        return img, target
    
    def _resize(self, img, boxes):
        """Resize image and boxes to target size"""
        orig_w, orig_h = img.size
        new_h, new_w = self.img_size
        
        # Resize image
        img = img.resize((new_w, new_h), Image.BILINEAR)
        
        # Scale boxes
        if len(boxes) > 0:
            scale_x = new_w / orig_w
            scale_y = new_h / orig_h
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
        
        return img, boxes
    
    def _augment(self, img, boxes):
        """Apply data augmentation"""
        # Random horizontal flip
        if random.random() > 0.5:
            img = TF.hflip(img)
            if len(boxes) > 0:
                w = img.size[0]
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        
        # Random color jitter
        if random.random() > 0.5:
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            saturation = random.uniform(0.8, 1.2)
            
            img = TF.adjust_brightness(img, brightness)
            img = TF.adjust_contrast(img, contrast)
            img = TF.adjust_saturation(img, saturation)
        
        return img, boxes


class SimpleDataset(Dataset):
    """
    Simple dataset for quick testing with image folder structure
    
    Expected structure:
        data_dir/
            images/
                img1.jpg
                img2.jpg
            labels/
                img1.txt
                img2.txt
    
    Label format (YOLO format):
        class_id x_center y_center width height (normalized)
    """
    
    def __init__(self, data_dir, img_size=640, augment=True):
        self.img_dir = os.path.join(data_dir, 'images')
        self.label_dir = os.path.join(data_dir, 'labels')
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.augment = augment
        
        # Get image files
        self.img_files = []
        for ext in ['jpg', 'jpeg', 'png', 'bmp']:
            self.img_files.extend([f for f in os.listdir(self.img_dir) if f.lower().endswith(ext)])
        
        print(f"Found {len(self.img_files)} images")
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # Load image
        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        
        # Load labels
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_file)
        
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * orig_w
                        y_center = float(parts[2]) * orig_h
                        w = float(parts[3]) * orig_w
                        h = float(parts[4]) * orig_h
                        
                        x1 = x_center - w / 2
                        y1 = y_center - h / 2
                        x2 = x_center + w / 2
                        y2 = y_center + h / 2
                        
                        boxes.append([x1, y1, x2, y2])
                        labels.append(class_id)
        
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        
        # Apply augmentations
        if self.augment:
            img, boxes = self._augment(img, boxes)
        
        # Resize
        img, boxes = self._resize(img, boxes, orig_w, orig_h)
        
        # To tensor and normalize
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'orig_size': torch.tensor([orig_h, orig_w])
        }
        
        return img, target
    
    def _resize(self, img, boxes, orig_w, orig_h):
        """Resize image and boxes"""
        new_h, new_w = self.img_size
        img = img.resize((new_w, new_h), Image.BILINEAR)
        
        if len(boxes) > 0:
            boxes[:, [0, 2]] *= new_w / orig_w
            boxes[:, [1, 3]] *= new_h / orig_h
        
        return img, boxes
    
    def _augment(self, img, boxes):
        """Apply augmentations"""
        if random.random() > 0.5:
            img = TF.hflip(img)
            if len(boxes) > 0:
                w = img.size[0]
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        return img, boxes


def collate_fn(batch):
    """Custom collate function for variable-sized targets"""
    images = []
    targets = []
    
    for img, target in batch:
        images.append(img)
        targets.append(target)
    
    images = torch.stack(images, dim=0)
    
    return images, targets


def create_dataloader(dataset, batch_size=16, shuffle=True, num_workers=4):
    """Create a DataLoader with proper collate function"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )


if __name__ == '__main__':
    # Test with dummy data
    print("Dataset module loaded successfully")

