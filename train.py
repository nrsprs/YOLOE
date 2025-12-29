"""
YOLOE Training Script
"""
import os
import sys
import time
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from models.yoloe import build_yoloe, YOLOE
from dataset import COCODataset, SimpleDataset, create_dataloader


def setup_logging(output_dir):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOE Training')
    
    # Data
    parser.add_argument('--data-dir', type=str, default='./data', help='Path to data directory')
    parser.add_argument('--train-ann', type=str, default=None, help='Path to training annotations (COCO format)')
    parser.add_argument('--val-ann', type=str, default=None, help='Path to validation annotations')
    
    # Model
    parser.add_argument('--model-size', type=str, default='s', choices=['s', 'm', 'l', 'x'],
                        help='Model size')
    parser.add_argument('--num-classes', type=int, default=80, help='Number of classes')
    parser.add_argument('--pretrained', type=str, default=None, help='Path to pretrained weights')
    
    # Training
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size per GPU')
    parser.add_argument('--img-size', type=int, default=640, help='Input image size')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.937, help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='Weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=3, help='Warmup epochs')
    
    # System
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--output-dir', type=str, default='./runs/train', help='Output directory')
    parser.add_argument('--save-period', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    return parser.parse_args()


class YOLOELoss(nn.Module):
    """
    YOLOE loss function
    Combines classification loss, box regression loss, and DFL loss
    """
    
    def __init__(self, num_classes=80, reg_max=16, strides=[32, 16, 8]):
        super(YOLOELoss, self).__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = strides
        
        # Loss weights
        self.cls_weight = 1.0
        self.box_weight = 2.5
        self.dfl_weight = 0.5
        
        # BCE for classification (sigmoid already applied in model)
        self.bce = nn.BCELoss(reduction='none')
        
    def forward(self, predictions, targets, anchor_points, stride_tensor):
        """
        Compute losses with improved positive/negative sample handling
        """
        cls_scores, reg_distri = predictions
        device = cls_scores.device
        batch_size = cls_scores.shape[0]
        num_anchors = cls_scores.shape[1]
        
        # Initialize losses
        loss_cls = torch.tensor(0.0, device=device)
        loss_box = torch.tensor(0.0, device=device)
        loss_dfl = torch.tensor(0.0, device=device)
        
        num_pos_total = 0
        
        for i in range(batch_size):
            gt_boxes = targets[i]['boxes'].to(device)
            gt_labels = targets[i]['labels'].to(device)
            
            pred_cls = cls_scores[i].clamp(1e-7, 1 - 1e-7)
            pred_reg = reg_distri[i]
            
            if len(gt_boxes) == 0:
                target_cls = torch.zeros_like(pred_cls)
                loss_cls += self.bce(pred_cls, target_cls).mean()
                continue
            
            num_gts = len(gt_boxes)
            gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2
            distances = torch.cdist(anchor_points, gt_centers)
            
            target_cls = torch.zeros_like(pred_cls)
            target_box = torch.zeros((num_anchors, 4), device=device)
            fg_mask = torch.zeros(num_anchors, dtype=torch.bool, device=device)
            
            # Multi-anchor assignment per GT
            k = min(13, num_anchors // max(num_gts, 1))
            k = max(k, 3)
            
            for gt_idx in range(num_gts):
                _, topk_indices = torch.topk(distances[:, gt_idx], k=k, largest=False)
                gt_box = gt_boxes[gt_idx]
                
                # Check anchors inside GT box
                inside = (
                    (anchor_points[topk_indices, 0] >= gt_box[0]) &
                    (anchor_points[topk_indices, 0] <= gt_box[2]) &
                    (anchor_points[topk_indices, 1] >= gt_box[1]) &
                    (anchor_points[topk_indices, 1] <= gt_box[3])
                )
                
                assigned = topk_indices[inside] if inside.any() else topk_indices[:3]
                
                for idx in assigned:
                    target_cls[idx, gt_labels[gt_idx]] = 1.0
                    target_box[idx] = gt_box
                    fg_mask[idx] = True
            
            num_pos = fg_mask.sum().item()
            num_pos_total += num_pos
            
            # Focal-like weighting for classification
            if num_pos > 0:
                pos_weight = min(num_anchors / num_pos, 50.0)
                pos_loss = self.bce(pred_cls[fg_mask], target_cls[fg_mask]).mean() * pos_weight
            else:
                pos_loss = torch.tensor(0.0, device=device)
            
            neg_loss = self.bce(pred_cls[~fg_mask], target_cls[~fg_mask]).mean()
            loss_cls += pos_loss + neg_loss
            
            if fg_mask.any():
                pred_boxes = self._decode_boxes(pred_reg[fg_mask], anchor_points[fg_mask], stride_tensor[fg_mask])
                loss_box += nn.functional.smooth_l1_loss(pred_boxes, target_box[fg_mask], reduction='mean')
        
        loss_cls = loss_cls / batch_size * self.cls_weight
        loss_box = loss_box / batch_size * self.box_weight
        
        return {
            'loss': loss_cls + loss_box,
            'loss_cls': loss_cls,
            'loss_box': loss_box,
            'loss_dfl': loss_dfl
        }
    
    def _decode_boxes(self, reg_distri, anchor_points, stride_tensor):
        """Decode regression distribution to boxes"""
        # Simplified decoding - in practice use DFL
        # reg_distri: (N, 4 * (reg_max + 1))
        # Reshape and take weighted sum
        reg_distri = reg_distri.reshape(-1, 4, self.reg_max + 1)
        reg_distri = nn.functional.softmax(reg_distri, dim=-1)
        
        proj = torch.linspace(0, self.reg_max, self.reg_max + 1, device=reg_distri.device)
        reg_dist = (reg_distri * proj).sum(dim=-1)  # (N, 4) - left, top, right, bottom
        
        # Decode to xyxy
        x1 = anchor_points[:, 0] - reg_dist[:, 0] * stride_tensor.squeeze()
        y1 = anchor_points[:, 1] - reg_dist[:, 1] * stride_tensor.squeeze()
        x2 = anchor_points[:, 0] + reg_dist[:, 2] * stride_tensor.squeeze()
        y2 = anchor_points[:, 1] + reg_dist[:, 3] * stride_tensor.squeeze()
        
        return torch.stack([x1, y1, x2, y2], dim=-1)


def train_one_epoch(model, dataloader, optimizer, loss_fn, device, epoch, scaler=None, logger=None):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    total_cls_loss = 0
    total_box_loss = 0
    num_batches = len(dataloader)
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        
        # Get feature map sizes for anchor generation
        with torch.no_grad():
            dummy_out = model(images[:1])
        
        # Generate anchor points
        # Assuming 3 feature levels with strides [32, 16, 8]
        h, w = images.shape[2:]
        feat_sizes = [(h // s, w // s) for s in [32, 16, 8]]
        anchor_points, stride_tensor = model.get_anchor_points(feat_sizes, device, images.dtype)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with autocast():
                predictions = model(images)
                loss_dict = loss_fn(predictions, targets, anchor_points, stride_tensor)
                loss = loss_dict['loss']
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            predictions = model(images)
            loss_dict = loss_fn(predictions, targets, anchor_points, stride_tensor)
            loss = loss_dict['loss']
            
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        total_cls_loss += loss_dict['loss_cls'].item()
        total_box_loss += loss_dict['loss_box'].item()
        
        if batch_idx % 10 == 0:
            msg = f"Epoch [{epoch}] Batch [{batch_idx}/{num_batches}] " \
                  f"Loss: {loss.item():.4f} " \
                  f"Cls: {loss_dict['loss_cls'].item():.4f} " \
                  f"Box: {loss_dict['loss_box'].item():.4f}"
            if logger:
                logger.info(msg)
            else:
                print(msg)
    
    avg_loss = total_loss / num_batches
    avg_cls = total_cls_loss / num_batches
    avg_box = total_box_loss / num_batches
    
    return {'loss': avg_loss, 'loss_cls': avg_cls, 'loss_box': avg_box}


def validate(model, dataloader, loss_fn, device):
    """Validate the model"""
    model.eval()
    
    total_loss = 0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            
            # Generate anchor points
            h, w = images.shape[2:]
            feat_sizes = [(h // s, w // s) for s in [32, 16, 8]]
            anchor_points, stride_tensor = model.get_anchor_points(feat_sizes, device, images.dtype)
            
            predictions = model(images)
            loss_dict = loss_fn(predictions, targets, anchor_points, stride_tensor)
            
            total_loss += loss_dict['loss'].item()
    
    return total_loss / num_batches


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save training checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info(f"Arguments: {args}")
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Build model
    logger.info(f"Building YOLOE-{args.model_size} model...")
    model = build_yoloe(args.model_size, args.num_classes)
    model = model.to(device)
    
    # Load pretrained weights if provided
    if args.pretrained:
        logger.info(f"Loading pretrained weights from {args.pretrained}")
        model.load_state_dict(torch.load(args.pretrained, map_location=device))
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create datasets
    if args.train_ann:
        train_dataset = COCODataset(
            img_dir=os.path.join(args.data_dir, 'train'),
            ann_file=args.train_ann,
            img_size=args.img_size,
            augment=True
        )
    else:
        train_dataset = SimpleDataset(
            data_dir=os.path.join(args.data_dir, 'train'),
            img_size=args.img_size,
            augment=True
        )
    
    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers
    )
    
    # Validation dataset (optional)
    val_loader = None
    if args.val_ann:
        val_dataset = COCODataset(
            img_dir=os.path.join(args.data_dir, 'val'),
            ann_file=args.val_ann,
            img_size=args.img_size,
            augment=False
        )
        val_loader = create_dataloader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers
        )
    
    # Loss function
    loss_fn = YOLOELoss(num_classes=args.num_classes)
    
    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if args.amp else None
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=output_dir / 'tensorboard')
    
    # Training loop
    logger.info("Starting training...")
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        # Warmup learning rate
        if epoch < args.warmup_epochs:
            warmup_lr = args.lr * (epoch + 1) / args.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        # Train one epoch
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn,
            device, epoch, scaler, logger
        )
        
        # Update learning rate
        if epoch >= args.warmup_epochs:
            scheduler.step()
        
        # Log metrics
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch} completed. Loss: {train_metrics['loss']:.4f}, LR: {current_lr:.6f}")
        
        writer.add_scalar('train/loss', train_metrics['loss'], epoch)
        writer.add_scalar('train/loss_cls', train_metrics['loss_cls'], epoch)
        writer.add_scalar('train/loss_box', train_metrics['loss_box'], epoch)
        writer.add_scalar('train/lr', current_lr, epoch)
        
        # Validation
        if val_loader is not None:
            val_loss = validate(model, val_loader, loss_fn, device)
            logger.info(f"Validation Loss: {val_loss:.4f}")
            writer.add_scalar('val/loss', val_loss, epoch)
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(
                    model, optimizer, epoch, val_loss,
                    output_dir / 'best.pt'
                )
                logger.info(f"Saved best model with loss: {best_loss:.4f}")
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_period == 0:
            save_checkpoint(
                model, optimizer, epoch, train_metrics['loss'],
                output_dir / f'epoch_{epoch}.pt'
            )
        
        # Save latest checkpoint
        save_checkpoint(
            model, optimizer, epoch, train_metrics['loss'],
            output_dir / 'last.pt'
        )
    
    # Save final model
    torch.save(model.state_dict(), output_dir / 'final.pt')
    logger.info(f"Training completed. Final model saved to {output_dir / 'final.pt'}")
    
    writer.close()


if __name__ == '__main__':
    main()

