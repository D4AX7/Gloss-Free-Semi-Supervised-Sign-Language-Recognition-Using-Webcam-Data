"""
Full Dataset Converter with Data Augmentation
Uses ALL 26,000 images to create ~25,000 training samples
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import json
from tqdm import tqdm
import random


class FullDatasetConverter:
    """Convert full A-Z dataset using all images with augmentation"""
    
    def __init__(self, frame_width=224, frame_height=224, frames_per_clip=16):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frames_per_clip = frames_per_clip
        
        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5
        )
    
    def augment_frame(self, frame, apply_augmentation=True):
        """Apply random augmentation to frame"""
        if not apply_augmentation:
            return frame
        
        # Random brightness (±30%)
        if random.random() > 0.5:
            brightness = random.uniform(0.7, 1.3)
            frame = np.clip(frame.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
        
        # Random contrast (±20%)
        if random.random() > 0.5:
            contrast = random.uniform(0.8, 1.2)
            mean = frame.mean()
            frame = np.clip((frame.astype(np.float32) - mean) * contrast + mean, 0, 255).astype(np.uint8)
        
        # Random small rotation (±10 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            h, w = frame.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        # Random crop and resize (90-100% of image)
        if random.random() > 0.5:
            h, w = frame.shape[:2]
            crop_ratio = random.uniform(0.9, 1.0)
            new_h, new_w = int(h * crop_ratio), int(w * crop_ratio)
            
            top = random.randint(0, h - new_h)
            left = random.randint(0, w - new_w)
            
            frame = frame[top:top+new_h, left:left+new_w]
            frame = cv2.resize(frame, (w, h))
        
        return frame
    
    def extract_landmarks(self, frame):
        """Extract hand and pose landmarks"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Hand landmarks
        hand_results = self.hands.process(rgb)
        hand_lms = []
        
        if hand_results.multi_hand_landmarks:
            for hand in hand_results.multi_hand_landmarks:
                for lm in hand.landmark:
                    hand_lms.extend([lm.x, lm.y, lm.z])
        
        # Pad to 2 hands (126 values)
        while len(hand_lms) < 126:
            hand_lms.append(0.0)
        hand_lms = hand_lms[:126]
        
        # Pose landmarks
        pose_results = self.pose.process(rgb)
        pose_lms = []
        
        if pose_results.pose_landmarks:
            for lm in pose_results.pose_landmarks.landmark:
                pose_lms.extend([lm.x, lm.y, lm.z, lm.visibility])
        
        if len(pose_lms) < 132:
            pose_lms = [0.0] * 132
        
        return np.array(hand_lms, dtype=np.float32), np.array(pose_lms, dtype=np.float32)
    
    def process_images_with_augmentation(self, image_paths, num_augmentations=2):
        """
        Process images and create multiple augmented versions
        
        Args:
            image_paths: List of image file paths
            num_augmentations: Number of augmented versions per sequence
        
        Returns:
            List of (frames, hand_landmarks, pose_landmarks) tuples
        """
        all_samples = []
        
        # Original version (no augmentation)
        frames, hand_lms, pose_lms = self.process_images(image_paths, apply_augmentation=False)
        if len(frames) > 0:
            all_samples.append((frames, hand_lms, pose_lms))
        
        # Augmented versions
        for _ in range(num_augmentations - 1):
            frames, hand_lms, pose_lms = self.process_images(image_paths, apply_augmentation=True)
            if len(frames) > 0:
                all_samples.append((frames, hand_lms, pose_lms))
        
        return all_samples
    
    def process_images(self, image_paths, apply_augmentation=False):
        """Process multiple images as a temporal sequence"""
        frames = []
        hand_landmarks_list = []
        pose_landmarks_list = []
        
        for img_path in image_paths:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            
            # Apply augmentation
            frame = self.augment_frame(frame, apply_augmentation)
            
            # Resize
            frame_resized = cv2.resize(frame, (self.frame_width, self.frame_height))
            frames.append(frame_resized)
            
            # Extract landmarks
            hand_lms, pose_lms = self.extract_landmarks(frame)
            hand_landmarks_list.append(hand_lms)
            pose_landmarks_list.append(pose_lms)
        
        # Temporal resize to fixed length
        if len(frames) > 0:
            frames = self.temporal_resize_frames(frames, self.frames_per_clip)
            hand_landmarks_list = self.temporal_resize_landmarks(hand_landmarks_list, self.frames_per_clip)
            pose_landmarks_list = self.temporal_resize_landmarks(pose_landmarks_list, self.frames_per_clip)
        
        return frames, hand_landmarks_list, pose_landmarks_list
    
    def temporal_resize_frames(self, frames, target_length):
        """Resize temporal dimension of frames using interpolation"""
        current_length = len(frames)
        if current_length == 0:
            return []
        
        if current_length == target_length:
            return frames
        
        # Linear interpolation
        indices = np.linspace(0, current_length - 1, target_length)
        resized_frames = []
        
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
            resized_frames.append(frame)
        
        return resized_frames
    
    def temporal_resize_landmarks(self, landmarks, target_length):
        """Resize landmark temporal dimension"""
        current_length = len(landmarks)
        if current_length == 0:
            return []
        
        if current_length == target_length:
            return landmarks
        
        indices = np.linspace(0, current_length - 1, target_length)
        resized = [landmarks[min(int(idx), current_length - 1)] for idx in indices]
        
        return resized
    
    def convert_full_dataset(self, input_dir, output_dir, images_per_sequence=16, 
                            sequences_overlap=8, num_augmentations=3):
        """
        Convert full dataset using ALL images
        
        Args:
            input_dir: Directory with class folders (A, B, C, ...)
            output_dir: Output directory for processed data
            images_per_sequence: Number of images per training sample
            sequences_overlap: Overlap between sequences (for sliding window)
            num_augmentations: Number of augmented versions per sequence
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get all class folders
        class_folders = [d for d in input_path.iterdir() if d.is_dir()]
        class_folders = sorted(class_folders, key=lambda x: x.name)
        
        # Label mapping
        label_mapping = {}
        total_samples = 0
        
        print(f"\n{'='*60}")
        print(f"FULL DATASET CONVERSION - Using ALL Images")
        print(f"{'='*60}")
        print(f"Images per sequence: {images_per_sequence}")
        print(f"Sequence overlap: {sequences_overlap}")
        print(f"Augmentations per sequence: {num_augmentations}")
        print(f"{'='*60}\n")
        
        for class_idx, class_folder in enumerate(tqdm(class_folders, desc="Processing classes")):
            label = class_folder.name
            label_mapping[label] = class_idx
            
            # Create output folder
            output_label_dir = output_path / label
            output_label_dir.mkdir(parents=True, exist_ok=True)
            
            class_samples = 0
            
            # Get all images from ALL *_frames folders  
            images = []
            
            # Find all folders ending with '_frames'
            frames_folders = [d for d in class_folder.iterdir() if d.is_dir() and d.name.endswith('_frames')]
            
            print(f"\n{label}: Found {len(frames_folders)} frames folders")
            
            for frames_dir in frames_folders:
                folder_images = []
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    folder_images.extend(list(frames_dir.glob(ext)))
                
                if folder_images:
                    folder_images = sorted(folder_images)
                    print(f"  {frames_dir.name}: {len(folder_images)} images")
                    images.extend(folder_images)
            
            if not images:
                print(f"  No images found in {class_folder.name}")
                continue
            
            images = sorted(images)
            print(f"  → Total: {len(images)} images across all frames folders")
            
            # Create sequences using sliding window
            stride = images_per_sequence - sequences_overlap
            sample_idx = 0
            
            for start_idx in range(0, len(images) - images_per_sequence + 1, stride):
                end_idx = start_idx + images_per_sequence
                image_group = images[start_idx:end_idx]
                
                # Create multiple augmented versions
                try:
                    samples = self.process_images_with_augmentation(
                        image_group, 
                        num_augmentations=num_augmentations
                    )
                    
                    for aug_idx, (frames, hand_lms, pose_lms) in enumerate(samples):
                        if len(frames) == 0:
                            continue
                        
                        frames_array = np.array(frames, dtype=np.uint8)
                        hand_lms_array = np.array(hand_lms, dtype=np.float32)
                        pose_lms_array = np.array(pose_lms, dtype=np.float32)
                        
                        # Save
                        sample_name = f'{label}_seq{sample_idx:04d}_aug{aug_idx}'
                        save_path = output_label_dir / f'{sample_name}.npz'
                        
                        np.savez_compressed(
                            save_path,
                            frames=frames_array,
                            hand_landmarks=hand_lms_array,
                            pose_landmarks=pose_lms_array,
                            label=label
                        )
                        class_samples += 1
                        total_samples += 1
                    
                    sample_idx += 1
                
                except Exception as e:
                    print(f"  Error processing sequence {sample_idx}: {e}")
                    continue
            
            print(f"  → Created {class_samples} samples for {label}")
        
        # Save label mapping
        label_map_path = output_path.parent / 'label_mapping.json'
        with open(label_map_path, 'w') as f:
            json.dump(label_mapping, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"CONVERSION COMPLETE!")
        print(f"{'='*60}")
        print(f"Total samples created: {total_samples}")
        print(f"Classes: {len(label_mapping)}")
        print(f"Average samples per class: {total_samples / len(label_mapping):.1f}")
        print(f"Label mapping saved to: {label_map_path}")
        print(f"{'='*60}\n")
        
        return total_samples


if __name__ == "__main__":
    converter = FullDatasetConverter()
    
    # YOUR DATASET PATH
    input_directory = r"C:\Users\KIIT\OneDrive\Documents\Desktop\ASL_dynamic"
    output_directory = "data/labeled"
    
    print("="*60)
    print("CONFIGURATION:")
    print(f"  Input: {input_directory}")
    print(f"  Output: {output_directory}")
    print(f"  Images per sequence: 16")
    print(f"  Sequence overlap: 8 (50%)")
    print(f"  Augmentations per sequence: 3")
    print(f"  Expected output: ~25,000-30,000 samples")
    print("="*60)
    print("\nStarting conversion in 3 seconds...")
    print("Press Ctrl+C to cancel\n")
    
    import time
    time.sleep(3)
    
    # Convert with full dataset
    total = converter.convert_full_dataset(
        input_dir=input_directory,
        output_dir=output_directory,
        images_per_sequence=16,      # 16 frames per sample
        sequences_overlap=8,          # 50% overlap between sequences
        num_augmentations=3           # 3x data augmentation
    )
    
    print(f"\n✅ Successfully created {total} training samples!")
    print(f"Ready to train with: python src/train.py")
