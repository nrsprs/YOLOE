"""
Download a tiny test dataset for YOLOE training verification
Uses a small subset from Open Images or generates synthetic data
"""
import os
import json
import random
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path


def create_synthetic_dataset(output_dir, num_train=50, num_val=10, img_size=640, num_classes=5):
    """
    Create a synthetic dataset with random shapes for testing
    
    Args:
        output_dir: Output directory
        num_train: Number of training images
        num_val: Number of validation images
        img_size: Image size
        num_classes: Number of classes
    """
    output_dir = Path(output_dir)
    
    # Create directories
    train_img_dir = output_dir / 'train' / 'images'
    train_lbl_dir = output_dir / 'train' / 'labels'
    val_img_dir = output_dir / 'val' / 'images'
    val_lbl_dir = output_dir / 'val' / 'labels'
    
    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    class_names = ['rectangle', 'circle', 'triangle', 'ellipse', 'polygon'][:num_classes]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    def generate_image(img_path, lbl_path):
        """Generate a single image with random shapes"""
        img = Image.new('RGB', (img_size, img_size), color=(128, 128, 128))
        draw = ImageDraw.Draw(img)
        
        # Add some background noise
        pixels = img.load()
        for i in range(img_size):
            for j in range(img_size):
                noise = random.randint(-20, 20)
                r, g, b = pixels[i, j]
                pixels[i, j] = (max(0, min(255, r + noise)),
                               max(0, min(255, g + noise)),
                               max(0, min(255, b + noise)))
        
        draw = ImageDraw.Draw(img)
        
        labels = []
        num_objects = random.randint(1, 5)
        
        for _ in range(num_objects):
            class_id = random.randint(0, num_classes - 1)
            color = colors[class_id]
            
            # Random box
            min_size = img_size // 10
            max_size = img_size // 3
            
            w = random.randint(min_size, max_size)
            h = random.randint(min_size, max_size)
            x1 = random.randint(0, img_size - w)
            y1 = random.randint(0, img_size - h)
            x2 = x1 + w
            y2 = y1 + h
            
            # Draw shape based on class
            if class_id == 0:  # Rectangle
                draw.rectangle([x1, y1, x2, y2], fill=color, outline=(0, 0, 0), width=2)
            elif class_id == 1:  # Circle
                draw.ellipse([x1, y1, x2, y2], fill=color, outline=(0, 0, 0), width=2)
            elif class_id == 2:  # Triangle
                points = [(x1 + w//2, y1), (x1, y2), (x2, y2)]
                draw.polygon(points, fill=color, outline=(0, 0, 0))
            elif class_id == 3:  # Ellipse
                draw.ellipse([x1, y1, x2, y2], fill=color, outline=(0, 0, 0), width=2)
            else:  # Polygon
                n_points = random.randint(5, 8)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                r = min(w, h) // 2
                points = []
                for i in range(n_points):
                    angle = 2 * np.pi * i / n_points
                    px = cx + int(r * np.cos(angle))
                    py = cy + int(r * np.sin(angle))
                    points.append((px, py))
                draw.polygon(points, fill=color, outline=(0, 0, 0))
            
            # YOLO format: class x_center y_center width height (normalized)
            x_center = (x1 + x2) / 2 / img_size
            y_center = (y1 + y2) / 2 / img_size
            width = w / img_size
            height = h / img_size
            
            labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # Save image
        img.save(img_path)
        
        # Save labels
        with open(lbl_path, 'w') as f:
            f.write('\n'.join(labels))
    
    # Generate training images
    print(f"Generating {num_train} training images...")
    for i in range(num_train):
        img_path = train_img_dir / f'train_{i:04d}.jpg'
        lbl_path = train_lbl_dir / f'train_{i:04d}.txt'
        generate_image(img_path, lbl_path)
    
    # Generate validation images
    print(f"Generating {num_val} validation images...")
    for i in range(num_val):
        img_path = val_img_dir / f'val_{i:04d}.jpg'
        lbl_path = val_lbl_dir / f'val_{i:04d}.txt'
        generate_image(img_path, lbl_path)
    
    # Create class names file
    with open(output_dir / 'classes.txt', 'w') as f:
        f.write('\n'.join(class_names))
    
    print(f"Dataset created at {output_dir}")
    print(f"  Training: {num_train} images")
    print(f"  Validation: {num_val} images")
    print(f"  Classes: {class_names}")
    
    return output_dir


def create_coco_format_dataset(output_dir, num_train=50, num_val=10, img_size=640, num_classes=5):
    """
    Create a synthetic dataset in COCO format
    """
    output_dir = Path(output_dir)
    
    # Create directories
    train_img_dir = output_dir / 'train'
    val_img_dir = output_dir / 'val'
    ann_dir = output_dir / 'annotations'
    
    for d in [train_img_dir, val_img_dir, ann_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    class_names = ['rectangle', 'circle', 'triangle', 'ellipse', 'polygon'][:num_classes]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    def generate_images_and_annotations(img_dir, num_images, start_id=0):
        """Generate images and return COCO annotations"""
        images = []
        annotations = []
        ann_id = 0
        
        for i in range(num_images):
            img_id = start_id + i
            filename = f'img_{img_id:04d}.jpg'
            
            img = Image.new('RGB', (img_size, img_size), color=(128, 128, 128))
            draw = ImageDraw.Draw(img)
            
            # Add background noise
            pixels = img.load()
            for x in range(img_size):
                for y in range(img_size):
                    noise = random.randint(-20, 20)
                    r, g, b = pixels[x, y]
                    pixels[x, y] = (max(0, min(255, r + noise)),
                                   max(0, min(255, g + noise)),
                                   max(0, min(255, b + noise)))
            
            draw = ImageDraw.Draw(img)
            
            images.append({
                'id': img_id,
                'file_name': filename,
                'width': img_size,
                'height': img_size
            })
            
            num_objects = random.randint(1, 5)
            
            for _ in range(num_objects):
                class_id = random.randint(0, num_classes - 1)
                color = colors[class_id]
                
                min_size = img_size // 10
                max_size = img_size // 3
                
                w = random.randint(min_size, max_size)
                h = random.randint(min_size, max_size)
                x1 = random.randint(0, img_size - w)
                y1 = random.randint(0, img_size - h)
                x2 = x1 + w
                y2 = y1 + h
                
                # Draw shape
                if class_id == 0:
                    draw.rectangle([x1, y1, x2, y2], fill=color, outline=(0, 0, 0), width=2)
                elif class_id == 1:
                    draw.ellipse([x1, y1, x2, y2], fill=color, outline=(0, 0, 0), width=2)
                elif class_id == 2:
                    points = [(x1 + w//2, y1), (x1, y2), (x2, y2)]
                    draw.polygon(points, fill=color, outline=(0, 0, 0))
                elif class_id == 3:
                    draw.ellipse([x1, y1, x2, y2], fill=color, outline=(0, 0, 0), width=2)
                else:
                    n_points = random.randint(5, 8)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    r = min(w, h) // 2
                    points = []
                    for j in range(n_points):
                        angle = 2 * np.pi * j / n_points
                        px = cx + int(r * np.cos(angle))
                        py = cy + int(r * np.sin(angle))
                        points.append((px, py))
                    draw.polygon(points, fill=color, outline=(0, 0, 0))
                
                # COCO format annotation
                annotations.append({
                    'id': ann_id,
                    'image_id': img_id,
                    'category_id': class_id + 1,  # COCO uses 1-indexed
                    'bbox': [x1, y1, w, h],  # COCO format: [x, y, width, height]
                    'area': w * h,
                    'iscrowd': 0
                })
                ann_id += 1
            
            # Save image
            img.save(img_dir / filename)
        
        return images, annotations
    
    # Categories
    categories = [{'id': i + 1, 'name': name} for i, name in enumerate(class_names)]
    
    # Generate training data
    print(f"Generating {num_train} training images...")
    train_images, train_anns = generate_images_and_annotations(train_img_dir, num_train, 0)
    
    train_coco = {
        'images': train_images,
        'annotations': train_anns,
        'categories': categories
    }
    
    with open(ann_dir / 'instances_train.json', 'w') as f:
        json.dump(train_coco, f)
    
    # Generate validation data
    print(f"Generating {num_val} validation images...")
    val_images, val_anns = generate_images_and_annotations(val_img_dir, num_val, num_train)
    
    val_coco = {
        'images': val_images,
        'annotations': val_anns,
        'categories': categories
    }
    
    with open(ann_dir / 'instances_val.json', 'w') as f:
        json.dump(val_coco, f)
    
    print(f"COCO format dataset created at {output_dir}")
    print(f"  Training: {num_train} images")
    print(f"  Validation: {num_val} images")
    print(f"  Classes: {class_names}")
    
    return output_dir


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='./test_data', help='Output directory')
    parser.add_argument('--format', type=str, default='yolo', choices=['yolo', 'coco'], help='Dataset format')
    parser.add_argument('--num-train', type=int, default=50, help='Number of training images')
    parser.add_argument('--num-val', type=int, default=10, help='Number of validation images')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--num-classes', type=int, default=5, help='Number of classes')
    
    args = parser.parse_args()
    
    if args.format == 'yolo':
        create_synthetic_dataset(args.output_dir, args.num_train, args.num_val, 
                                 args.img_size, args.num_classes)
    else:
        create_coco_format_dataset(args.output_dir, args.num_train, args.num_val,
                                   args.img_size, args.num_classes)

