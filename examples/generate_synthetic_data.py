import numpy as np
import cv2
import argparse
from pathlib import Path
import yaml
import json


def generate_synthetic_data(config_path='configs/config.yaml', output_dir='data', num_samples_per_class=20):
    """
    Generate synthetic data for testing the pipeline
    This creates random videos with random landmarks to test the system
    """
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    frame_height = config['data']['frame_height']
    frame_width = config['data']['frame_width']
    frames_per_clip = config['data']['frames_per_clip']
    
    # Define some example sign labels
    labels = ['hello', 'thanks', 'yes', 'no', 'please', 'sorry', 'help', 'stop', 'go', 'wait']
    
    # Create label mapping
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    
    # Update config with correct number of classes
    config['model']['num_classes'] = len(labels)
    
    print(f"Generating synthetic data with {len(labels)} classes...")
    print(f"Classes: {labels}")
    print(f"Samples per class: {num_samples_per_class}")
    
    # Generate labeled data
    labeled_dir = Path(output_dir) / 'labeled'
    for label in labels:
        label_dir = labeled_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(num_samples_per_class):
            # Generate random frames with some pattern
            frames = []
            for t in range(frames_per_clip):
                # Create a pattern that varies over time
                # This makes different signs look slightly different
                pattern = label_to_idx[label] * 25
                color_shift = int(255 * t / frames_per_clip)
                
                frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                frame[:, :, 0] = (pattern + color_shift) % 256  # R channel
                frame[:, :, 1] = (pattern * 2) % 256  # G channel
                frame[:, :, 2] = (255 - pattern) % 256  # B channel
                
                # Add some random noise
                noise = np.random.randint(0, 50, frame.shape, dtype=np.uint8)
                frame = cv2.add(frame, noise)
                
                frames.append(frame)
            
            frames = np.array(frames, dtype=np.uint8)
            
            # Generate random hand landmarks
            # 2 hands, 21 landmarks per hand, 3 coordinates per landmark
            hand_landmarks = np.random.rand(frames_per_clip, 2, 63).astype(np.float32)
            
            # Add some temporal consistency
            for t in range(1, frames_per_clip):
                hand_landmarks[t] = 0.7 * hand_landmarks[t] + 0.3 * hand_landmarks[t-1]
            
            # Generate random pose landmarks
            # 33 landmarks, 4 values per landmark (x, y, z, visibility)
            pose_landmarks = np.random.rand(frames_per_clip, 132).astype(np.float32)
            
            # Add temporal consistency
            for t in range(1, frames_per_clip):
                pose_landmarks[t] = 0.7 * pose_landmarks[t] + 0.3 * pose_landmarks[t-1]
            
            # Save
            save_path = label_dir / f'sample_{i:04d}.npz'
            np.savez_compressed(
                save_path,
                frames=frames,
                hand_landmarks=hand_landmarks,
                pose_landmarks=pose_landmarks,
                label=label
            )
    
    print(f"Generated {num_samples_per_class * len(labels)} labeled samples")
    
    # Generate unlabeled data (fewer samples)
    unlabeled_dir = Path(output_dir) / 'unlabeled' / 'unknown'
    unlabeled_dir.mkdir(parents=True, exist_ok=True)
    
    num_unlabeled = num_samples_per_class * len(labels) // 2
    
    for i in range(num_unlabeled):
        # Similar to labeled data but with random patterns
        frames = []
        for t in range(frames_per_clip):
            frame = np.random.randint(0, 256, (frame_height, frame_width, 3), dtype=np.uint8)
            frames.append(frame)
        
        frames = np.array(frames, dtype=np.uint8)
        hand_landmarks = np.random.rand(frames_per_clip, 2, 63).astype(np.float32)
        pose_landmarks = np.random.rand(frames_per_clip, 132).astype(np.float32)
        
        # Add temporal consistency
        for t in range(1, frames_per_clip):
            hand_landmarks[t] = 0.7 * hand_landmarks[t] + 0.3 * hand_landmarks[t-1]
            pose_landmarks[t] = 0.7 * pose_landmarks[t] + 0.3 * pose_landmarks[t-1]
        
        save_path = unlabeled_dir / f'sample_{i:04d}.npz'
        np.savez_compressed(
            save_path,
            frames=frames,
            hand_landmarks=hand_landmarks,
            pose_landmarks=pose_landmarks,
            label='unknown'
        )
    
    print(f"Generated {num_unlabeled} unlabeled samples")
    
    # Save label mapping
    label_map_path = Path(output_dir) / 'label_mapping.json'
    with open(label_map_path, 'w') as f:
        json.dump(label_to_idx, f, indent=2)
    
    print(f"Saved label mapping to {label_map_path}")
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nSynthetic data generation complete!")
    print(f"Labeled data: {labeled_dir}")
    print(f"Unlabeled data: {unlabeled_dir}")
    print(f"\nNext steps:")
    print(f"1. Run: python src/preprocess.py")
    print(f"2. Run: python src/train.py")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic sign language data')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Config file')
    parser.add_argument('--output', type=str, default='data', help='Output directory')
    parser.add_argument('--samples', type=int, default=20, help='Samples per class')
    
    args = parser.parse_args()
    
    generate_synthetic_data(args.config, args.output, args.samples)


if __name__ == '__main__':
    main()
