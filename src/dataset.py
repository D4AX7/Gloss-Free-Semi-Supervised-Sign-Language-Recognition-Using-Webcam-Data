import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path


class SignLanguageDataset(Dataset):
    """PyTorch Dataset for Sign Language Recognition"""
    
    def __init__(self, data_path, label_mapping_path=None, transform=None, is_labeled=True):
        """
        Args:
            data_path: Path to .npz file containing preprocessed data
            label_mapping_path: Path to label mapping JSON file
            transform: Optional transform to apply
            is_labeled: Whether dataset has labels
        """
        self.data = np.load(data_path, allow_pickle=True)
        self.transform = transform
        self.is_labeled = is_labeled
        
        self.frames = self.data['frames']
        self.hand_landmarks = self.data['hand_landmarks']
        self.pose_landmarks = self.data['pose_landmarks']
        
        if is_labeled:
            self.labels = self.data['labels']
        else:
            self.labels = None
        
        # Load label mapping if provided
        self.label_mapping = None
        self.idx_to_label = None
        if label_mapping_path and Path(label_mapping_path).exists():
            with open(label_mapping_path, 'r') as f:
                self.label_mapping = json.load(f)
                self.idx_to_label = {v: k for k, v in self.label_mapping.items()}
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        # Get data
        frames = self.frames[idx]  # (T, H, W, C)
        hand_lms = self.hand_landmarks[idx]  # (T, 2, 63)
        pose_lms = self.pose_landmarks[idx]  # (T, 132)
        
        # Convert to torch tensors
        frames = torch.from_numpy(frames).float()
        hand_lms = torch.from_numpy(hand_lms).float()
        pose_lms = torch.from_numpy(pose_lms).float()
        
        # Permute frames to (C, T, H, W) for 3D conv
        frames = frames.permute(3, 0, 1, 2)
        
        # Flatten hand landmarks
        hand_lms = hand_lms.reshape(hand_lms.shape[0], -1)
        
        # Apply transforms if any
        if self.transform:
            frames = self.transform(frames)
        
        if self.is_labeled:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return {
                'frames': frames,
                'hand_landmarks': hand_lms,
                'pose_landmarks': pose_lms,
                'label': label,
                'idx': idx
            }
        else:
            return {
                'frames': frames,
                'hand_landmarks': hand_lms,
                'pose_landmarks': pose_lms,
                'idx': idx
            }
    
    def get_num_classes(self):
        """Get number of unique classes"""
        if self.is_labeled and self.labels is not None:
            return len(np.unique(self.labels))
        elif self.label_mapping:
            return len(self.label_mapping)
        else:
            return 0


class PseudoLabeledDataset(Dataset):
    """Dataset that combines labeled and pseudo-labeled data"""
    
    def __init__(self, labeled_dataset, unlabeled_dataset, pseudo_labels, pseudo_confidences, threshold=0.9):
        """
        Args:
            labeled_dataset: Original labeled dataset
            unlabeled_dataset: Unlabeled dataset
            pseudo_labels: Predicted labels for unlabeled data
            pseudo_confidences: Confidence scores for predictions
            threshold: Minimum confidence to include pseudo-labeled sample
        """
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.pseudo_labels = pseudo_labels
        self.pseudo_confidences = pseudo_confidences
        self.threshold = threshold
        
        # Find high-confidence pseudo-labeled samples
        self.pseudo_mask = pseudo_confidences >= threshold
        self.pseudo_indices = np.where(self.pseudo_mask)[0]
        
        self.num_labeled = len(labeled_dataset)
        self.num_pseudo = len(self.pseudo_indices)
        
        print(f"Combined dataset: {self.num_labeled} labeled + {self.num_pseudo} pseudo-labeled")
    
    def __len__(self):
        return self.num_labeled + self.num_pseudo
    
    def __getitem__(self, idx):
        if idx < self.num_labeled:
            # Return labeled sample
            return self.labeled_dataset[idx]
        else:
            # Return pseudo-labeled sample
            pseudo_idx = self.pseudo_indices[idx - self.num_labeled]
            sample = self.unlabeled_dataset[pseudo_idx]
            
            # Add pseudo label
            sample['label'] = torch.tensor(self.pseudo_labels[pseudo_idx], dtype=torch.long)
            sample['is_pseudo'] = True
            
            return sample


def get_dataloaders(config, labeled_path, unlabeled_path=None, label_mapping_path=None, 
                   pseudo_labels=None, pseudo_confidences=None):
    """Create dataloaders for training"""
    
    # Create labeled dataset
    labeled_dataset = SignLanguageDataset(
        labeled_path,
        label_mapping_path=label_mapping_path,
        is_labeled=True
    )
    
    # Split into train/val
    train_size = int(0.8 * len(labeled_dataset))
    val_size = len(labeled_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        labeled_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create unlabeled dataset if provided
    if unlabeled_path and Path(unlabeled_path).exists():
        unlabeled_dataset = SignLanguageDataset(
            unlabeled_path,
            label_mapping_path=label_mapping_path,
            is_labeled=False
        )
        
        # If pseudo labels provided, create combined dataset
        if pseudo_labels is not None and pseudo_confidences is not None:
            threshold = config['semi_supervised']['pseudo_label_threshold']
            train_dataset = PseudoLabeledDataset(
                train_dataset, unlabeled_dataset,
                pseudo_labels, pseudo_confidences, threshold
            )
    
    # Create dataloaders
    batch_size = config['training']['batch_size']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader, labeled_dataset.get_num_classes()


if __name__ == '__main__':
    # Test dataset loading
    import yaml
    
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Try to load labeled data
    labeled_path = 'data/processed_labeled.npz'
    if Path(labeled_path).exists():
        dataset = SignLanguageDataset(labeled_path, is_labeled=True)
        print(f"Dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Frames shape: {sample['frames'].shape}")
            print(f"Hand landmarks shape: {sample['hand_landmarks'].shape}")
            print(f"Pose landmarks shape: {sample['pose_landmarks'].shape}")
            print(f"Label: {sample['label']}")
    else:
        print(f"No data found at {labeled_path}")
