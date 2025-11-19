import numpy as np
import cv2
import os
import yaml
from pathlib import Path
from tqdm import tqdm
import json


class DataPreprocessor:
    def __init__(self, config_path='configs/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.frame_height = self.config['data']['frame_height']
        self.frame_width = self.config['data']['frame_width']
        self.frames_per_clip = self.config['data']['frames_per_clip']
    
    def normalize_frames(self, frames):
        """Normalize frames to [0, 1]"""
        return frames.astype(np.float32) / 255.0
    
    def normalize_landmarks(self, landmarks):
        """Normalize landmarks (already in [0, 1] range mostly)"""
        return landmarks.astype(np.float32)
    
    def temporal_resize(self, frames, landmarks_hand, landmarks_pose, target_length):
        """Resize temporal dimension to fixed length"""
        current_length = len(frames)
        
        if current_length == target_length:
            return frames, landmarks_hand, landmarks_pose
        
        # Create indices for interpolation
        indices = np.linspace(0, current_length - 1, target_length)
        
        # Interpolate frames
        frames_resized = []
        for idx in indices:
            idx_int = int(idx)
            if idx_int < current_length - 1:
                alpha = idx - idx_int
                frame = cv2.addWeighted(
                    frames[idx_int], 1 - alpha,
                    frames[idx_int + 1], alpha, 0
                )
            else:
                frame = frames[idx_int]
            frames_resized.append(frame)
        
        # Interpolate landmarks
        hand_resized = np.array([
            landmarks_hand[min(int(idx), current_length - 1)]
            for idx in indices
        ])
        
        pose_resized = np.array([
            landmarks_pose[min(int(idx), current_length - 1)]
            for idx in indices
        ])
        
        return np.array(frames_resized), hand_resized, pose_resized
    
    def preprocess_sample(self, sample_path):
        """Preprocess a single sample"""
        data = np.load(sample_path, allow_pickle=True)
        
        frames = data['frames']
        hand_landmarks = data['hand_landmarks']
        pose_landmarks = data['pose_landmarks']
        label = str(data['label'])
        
        # Temporal resize
        frames, hand_landmarks, pose_landmarks = self.temporal_resize(
            frames, hand_landmarks, pose_landmarks, self.frames_per_clip
        )
        
        # Spatial resize (if needed)
        if frames.shape[1] != self.frame_height or frames.shape[2] != self.frame_width:
            frames_resized = []
            for frame in frames:
                frame_resized = cv2.resize(
                    frame, (self.frame_width, self.frame_height)
                )
                frames_resized.append(frame_resized)
            frames = np.array(frames_resized)
        
        # Normalize
        frames = self.normalize_frames(frames)
        hand_landmarks = self.normalize_landmarks(hand_landmarks)
        pose_landmarks = self.normalize_landmarks(pose_landmarks)
        
        return {
            'frames': frames,
            'hand_landmarks': hand_landmarks,
            'pose_landmarks': pose_landmarks,
            'label': label
        }
    
    def process_dataset(self, input_dir, output_file):
        """Process entire dataset"""
        input_path = Path(input_dir)
        samples = []
        labels_set = set()
        
        # Find all .npz files
        npz_files = list(input_path.rglob('*.npz'))
        
        if len(npz_files) == 0:
            print(f"No .npz files found in {input_dir}")
            return
        
        print(f"Processing {len(npz_files)} samples from {input_dir}")
        
        for sample_path in tqdm(npz_files, desc="Preprocessing"):
            try:
                processed = self.preprocess_sample(sample_path)
                samples.append(processed)
                labels_set.add(processed['label'])
            except Exception as e:
                print(f"Error processing {sample_path}: {e}")
                continue
        
        # Create label mapping
        label_to_idx = {label: idx for idx, label in enumerate(sorted(labels_set))}
        
        # Convert labels to indices
        for sample in samples:
            sample['label_idx'] = label_to_idx[sample['label']]
        
        # Save processed data
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as single file
        frames_list = [s['frames'] for s in samples]
        hand_lms_list = [s['hand_landmarks'] for s in samples]
        pose_lms_list = [s['pose_landmarks'] for s in samples]
        labels_list = [s['label_idx'] for s in samples]
        
        np.savez_compressed(
            output_path,
            frames=np.array(frames_list),
            hand_landmarks=np.array(hand_lms_list),
            pose_landmarks=np.array(pose_lms_list),
            labels=np.array(labels_list)
        )
        
        # Save label mapping
        label_map_path = output_path.parent / 'label_mapping.json'
        with open(label_map_path, 'w') as f:
            json.dump(label_to_idx, f, indent=2)
        
        print(f"Saved {len(samples)} processed samples to {output_path}")
        print(f"Label mapping saved to {label_map_path}")
        print(f"Labels: {sorted(labels_set)}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess sign language data')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Config file')
    parser.add_argument('--labeled_input', type=str, default='data/labeled', help='Labeled data input directory')
    parser.add_argument('--unlabeled_input', type=str, default='data/unlabeled', help='Unlabeled data input directory')
    parser.add_argument('--labeled_output', type=str, default='data/processed_labeled.npz', help='Labeled data output file')
    parser.add_argument('--unlabeled_output', type=str, default='data/processed_unlabeled.npz', help='Unlabeled data output file')
    
    args = parser.parse_args()
    
    preprocessor = DataPreprocessor(args.config)
    
    # Process labeled data
    if os.path.exists(args.labeled_input):
        print("\n=== Processing Labeled Data ===")
        preprocessor.process_dataset(args.labeled_input, args.labeled_output)
    
    # Process unlabeled data
    if os.path.exists(args.unlabeled_input):
        print("\n=== Processing Unlabeled Data ===")
        preprocessor.process_dataset(args.unlabeled_input, args.unlabeled_output)


if __name__ == '__main__':
    main()
