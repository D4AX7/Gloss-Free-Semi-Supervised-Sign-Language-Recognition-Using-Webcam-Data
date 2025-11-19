import torch
import yaml
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

from model import create_model
from dataset import SignLanguageDataset
from utils import load_checkpoint


class PseudoLabeler:
    def __init__(self, config, model_path, device='cuda'):
        self.config = config
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Load model
        self.model = create_model(config).to(self.device)
        checkpoint = load_checkpoint(model_path, self.model)
        self.model.eval()
        
        print(f"Loaded model from {model_path}")
        print(f"Using device: {self.device}")
    
    def generate_pseudo_labels(self, unlabeled_loader):
        """Generate pseudo labels for unlabeled data"""
        all_predictions = []
        all_confidences = []
        
        print("Generating pseudo labels...")
        
        with torch.no_grad():
            for batch in tqdm(unlabeled_loader, desc='Pseudo-labeling'):
                frames = batch['frames'].to(self.device)
                hand_lms = batch['hand_landmarks'].to(self.device)
                pose_lms = batch['pose_landmarks'].to(self.device)
                
                # Forward pass
                logits = self.model(frames, hand_lms, pose_lms)
                
                # Get predictions and confidences
                probs = torch.softmax(logits, dim=1)
                confidences, predictions = torch.max(probs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_confidences)
    
    def save_pseudo_labels(self, predictions, confidences, output_path):
        """Save pseudo labels to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(
            output_path,
            predictions=predictions,
            confidences=confidences
        )
        
        print(f"Saved pseudo labels to {output_path}")
        
        # Print statistics
        threshold = self.config['semi_supervised']['pseudo_label_threshold']
        high_conf_mask = confidences >= threshold
        num_high_conf = np.sum(high_conf_mask)
        
        print(f"\nPseudo-label Statistics:")
        print(f"  Total samples: {len(predictions)}")
        print(f"  High confidence (>={threshold}): {num_high_conf} ({100*num_high_conf/len(predictions):.1f}%)")
        print(f"  Mean confidence: {np.mean(confidences):.3f}")
        print(f"  Std confidence: {np.std(confidences):.3f}")
        
        # Per-class statistics
        unique_labels = np.unique(predictions[high_conf_mask])
        print(f"\nHigh-confidence samples per class:")
        for label in unique_labels:
            count = np.sum((predictions == label) & high_conf_mask)
            print(f"  Class {label}: {count} samples")


def main():
    parser = argparse.ArgumentParser(description='Generate pseudo labels for unlabeled data')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Config file')
    parser.add_argument('--model', type=str, default='models/best_model.pth', help='Trained model path')
    parser.add_argument('--unlabeled_data', type=str, default='data/processed_unlabeled.npz', help='Unlabeled data path')
    parser.add_argument('--label_mapping', type=str, default='data/label_mapping.json', help='Label mapping path')
    parser.add_argument('--output', type=str, default='data/pseudo_labels.npz', help='Output path for pseudo labels')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check if files exist
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        print("Please train a model first using train.py")
        return
    
    if not Path(args.unlabeled_data).exists():
        print(f"Error: Unlabeled data not found at {args.unlabeled_data}")
        print("Please preprocess unlabeled data first")
        return
    
    # Load unlabeled dataset
    print("Loading unlabeled data...")
    unlabeled_dataset = SignLanguageDataset(
        args.unlabeled_data,
        label_mapping_path=args.label_mapping if Path(args.label_mapping).exists() else None,
        is_labeled=False
    )
    
    unlabeled_loader = torch.utils.data.DataLoader(
        unlabeled_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"Loaded {len(unlabeled_dataset)} unlabeled samples")
    
    # Create pseudo labeler
    pseudo_labeler = PseudoLabeler(config, args.model, device=args.device)
    
    # Generate pseudo labels
    predictions, confidences = pseudo_labeler.generate_pseudo_labels(unlabeled_loader)
    
    # Save pseudo labels
    pseudo_labeler.save_pseudo_labels(predictions, confidences, args.output)
    
    print("\nPseudo-labeling complete!")
    print("You can now retrain the model with pseudo-labeled data by modifying train.py")


if __name__ == '__main__':
    main()
