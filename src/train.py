import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import os

from model import create_model
from dataset import get_dataloaders
from utils import AverageMeter, accuracy, save_checkpoint, load_checkpoint


class Trainer:
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        print(f"Using device: {self.device}")
        
        # Create model
        self.model = create_model(config).to(self.device)
        
        # Loss function
        label_smoothing = config['training'].get('label_smoothing', 0.0)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # Optimizer
        lr = config['training']['learning_rate']
        weight_decay = config['training']['weight_decay']
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['epochs']
        )
        
        # Mixed precision training
        self.use_amp = config['training'].get('mixed_precision', False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Tensorboard
        if config['logging']['tensorboard']:
            log_dir = Path(config['paths']['logs']) / 'train'
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
        
        self.best_acc = 0.0
        self.start_epoch = 0
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        
        losses = AverageMeter()
        top1 = AverageMeter()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            frames = batch['frames'].to(self.device)
            hand_lms = batch['hand_landmarks'].to(self.device)
            pose_lms = batch['pose_landmarks'].to(self.device)
            labels = batch['label'].to(self.device)
            
            batch_size = frames.size(0)
            
            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logits = self.model(frames, hand_lms, pose_lms)
                    loss = self.criterion(logits, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config['training'].get('gradient_clip', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(frames, hand_lms, pose_lms)
                loss = self.criterion(logits, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.config['training'].get('gradient_clip', 0) > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                self.optimizer.step()
            
            # Metrics
            acc1 = accuracy(logits, labels, topk=(1,))[0]
            losses.update(loss.item(), batch_size)
            top1.update(acc1.item(), batch_size)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{top1.avg:.2f}%'
            })
            
            # Log to tensorboard
            if self.writer and batch_idx % self.config['logging']['log_interval'] == 0:
                global_step = epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('train/loss', loss.item(), global_step)
                self.writer.add_scalar('train/acc', acc1.item(), global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], global_step)
        
        return losses.avg, top1.avg
    
    def validate(self, val_loader, epoch):
        """Validate model"""
        self.model.eval()
        
        losses = AverageMeter()
        top1 = AverageMeter()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                frames = batch['frames'].to(self.device)
                hand_lms = batch['hand_landmarks'].to(self.device)
                pose_lms = batch['pose_landmarks'].to(self.device)
                labels = batch['label'].to(self.device)
                
                batch_size = frames.size(0)
                
                # Forward pass
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        logits = self.model(frames, hand_lms, pose_lms)
                        loss = self.criterion(logits, labels)
                else:
                    logits = self.model(frames, hand_lms, pose_lms)
                    loss = self.criterion(logits, labels)
                
                # Metrics
                acc1 = accuracy(logits, labels, topk=(1,))[0]
                losses.update(loss.item(), batch_size)
                top1.update(acc1.item(), batch_size)
        
        # Log to tensorboard
        if self.writer:
            self.writer.add_scalar('val/loss', losses.avg, epoch)
            self.writer.add_scalar('val/acc', top1.avg, epoch)
        
        return losses.avg, top1.avg
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        epochs = self.config['training']['epochs']
        
        print(f"\nStarting training for {epochs} epochs...")
        
        for epoch in range(self.start_epoch, epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, epoch)
            
            # Update learning rate
            self.scheduler.step()
            
            # Print epoch summary
            print(f'\nEpoch {epoch}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'  LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # Save checkpoint
            is_best = val_acc > self.best_acc
            self.best_acc = max(val_acc, self.best_acc)
            
            if epoch % self.config['logging']['checkpoint_interval'] == 0 or is_best:
                checkpoint_dir = Path(self.config['paths']['models'])
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'best_acc': self.best_acc,
                        'config': self.config
                    },
                    is_best,
                    checkpoint_dir
                )
        
        print(f'\nTraining completed! Best accuracy: {self.best_acc:.2f}%')
        
        if self.writer:
            self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train Sign Language Recognition Model')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Config file')
    parser.add_argument('--labeled_data', type=str, default='data/processed_labeled.npz', help='Labeled data path')
    parser.add_argument('--unlabeled_data', type=str, default='data/processed_unlabeled.npz', help='Unlabeled data path')
    parser.add_argument('--label_mapping', type=str, default='data/label_mapping.json', help='Label mapping path')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check if data exists
    if not Path(args.labeled_data).exists():
        print(f"Error: Labeled data not found at {args.labeled_data}")
        print("Please run preprocess.py first or generate synthetic data")
        return
    
    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader, num_classes = get_dataloaders(
        config,
        args.labeled_data,
        args.unlabeled_data if Path(args.unlabeled_data).exists() else None,
        args.label_mapping if Path(args.label_mapping).exists() else None
    )
    
    # Update num_classes in config
    config['model']['num_classes'] = num_classes
    print(f"Number of classes: {num_classes}")
    
    # Create trainer
    trainer = Trainer(config, device=args.device)
    
    # Resume from checkpoint if specified
    if args.resume and Path(args.resume).exists():
        print(f"Resuming from {args.resume}")
        checkpoint = load_checkpoint(args.resume, trainer.model, trainer.optimizer, trainer.scheduler)
        trainer.start_epoch = checkpoint['epoch']
        trainer.best_acc = checkpoint.get('best_acc', 0.0)
    
    # Train
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
