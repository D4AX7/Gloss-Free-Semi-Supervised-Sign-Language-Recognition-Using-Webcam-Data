"""
Custom data loader for A-Z sign language dataset
Handles both images and videos, extracts landmarks using MediaPipe
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import yaml
import argparse
from tqdm import tqdm
import json


class AZDatasetConverter:
    def __init__(self, config_path='configs/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.frame_height = self.config['data']['frame_height']
        self.frame_width = self.config['data']['frame_width']
        self.frames_per_clip = self.config['data']['frames_per_clip']
    
    def extract_landmarks(self, frame):
        """Extract hand and pose landmarks from frame"""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hands
        hand_results = self.hands.process(image_rgb)
        hand_landmarks = []
        
        if hand_results.multi_hand_landmarks:
            for hand_lms in hand_results.multi_hand_landmarks:
                lms = []
                for lm in hand_lms.landmark:
                    lms.extend([lm.x, lm.y, lm.z])
                hand_landmarks.append(lms)
        
        # Pad to 2 hands
        while len(hand_landmarks) < 2:
            hand_landmarks.append([0.0] * 63)
        
        hand_landmarks = np.array(hand_landmarks, dtype=np.float32)
        
        # Process pose
        pose_results = self.pose.process(image_rgb)
        pose_landmarks = []
        
        if pose_results.pose_landmarks:
            for lm in pose_results.pose_landmarks.landmark:
                pose_landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            pose_landmarks = [0.0] * 132
        
        pose_landmarks = np.array(pose_landmarks, dtype=np.float32)
        
        return hand_landmarks, pose_landmarks
    
    def process_video(self, video_path):
        """Process a video file into frames and landmarks"""
        cap = cv2.VideoCapture(str(video_path))
        
        frames = []
        hand_landmarks_list = []
        pose_landmarks_list = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame
            frame_resized = cv2.resize(frame, (self.frame_width, self.frame_height))
            frames.append(frame_resized)
            
            # Extract landmarks
            hand_lms, pose_lms = self.extract_landmarks(frame)
            hand_landmarks_list.append(hand_lms)
            pose_landmarks_list.append(pose_lms)
        
        cap.release()
        
        # Temporal resize to fixed length
        if len(frames) > 0:
            frames = self.temporal_resize_frames(frames, self.frames_per_clip)
            hand_landmarks_list = self.temporal_resize_landmarks(hand_landmarks_list, self.frames_per_clip)
            pose_landmarks_list = self.temporal_resize_landmarks(pose_landmarks_list, self.frames_per_clip)
        
        return frames, hand_landmarks_list, pose_landmarks_list
    
    def process_images(self, image_paths):
        """Process multiple images as a sequence"""
        frames = []
        hand_landmarks_list = []
        pose_landmarks_list = []
        
        for img_path in image_paths:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            
            # Resize frame
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
        """Resize temporal dimension using interpolation"""
        current_length = len(frames)
        if current_length == 0:
            return []
        
        if current_length == target_length:
            return frames
        
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
    
    def convert_dataset(self, input_dir, output_dir, split='labeled', use_videos=True, use_images=True):
        """
        Convert A-Z + gestures dataset to project format
        
        Expected input structure:
        input_dir/
            A/
                image1.jpg, image2.jpg, ... (1000 images)
                OR video1.mp4 + frames/
            B/
                ...
            ...
            Z/
                ...
            Hello/
                video1.mp4 + frames/
                video2.mp4 + frames/
            Thank You/
                ...
            Sorry/
                ...
            Yes/
                ...
            No/
                ...
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir) / split
        
        if not input_path.exists():
            print(f"Error: Input directory not found: {input_dir}")
            return
        
        # Find all class directories (A-Z + gestures)
        class_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])
        
        print(f"Found {len(class_dirs)} classes")
        print(f"Converting dataset from {input_dir} to {output_path}")
        print(f"Processing videos: {use_videos}, Processing images: {use_images}")
        
        total_samples = 0
        class_stats = {}
        
        for class_dir in tqdm(class_dirs, desc="Processing classes"):
            label = class_dir.name
            output_label_dir = output_path / label
            output_label_dir.mkdir(parents=True, exist_ok=True)
            
            class_samples = 0
            
            # Find all video and image files
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            
            videos = []
            images = []
            frames_dirs = []
            
            # Scan the class directory
            for file in class_dir.iterdir():
                if file.is_file():
                    # Videos in main class folder
                    if file.suffix.lower() in video_extensions and use_videos:
                        videos.append(file)
                    # Images in main class folder (if any)
                    elif file.suffix.lower() in image_extensions and use_images:
                        images.append(file)
                elif file.is_dir():
                    # Check if this is a frames directory
                    dir_name_lower = file.name.lower()
                    if 'frame' in dir_name_lower or dir_name_lower in ['frames', 'frame', 'images', 'imgs']:
                        frames_dirs.append(file)
            
            # Process videos
            if use_videos:
                for video_file in videos:
                    try:
                        frames, hand_lms, pose_lms = self.process_video(video_file)
                        
                        if len(frames) > 0:
                            frames_array = np.array(frames, dtype=np.uint8)
                            hand_lms_array = np.array(hand_lms, dtype=np.float32)
                            pose_lms_array = np.array(pose_lms, dtype=np.float32)
                            
                            # Save
                            sample_name = video_file.stem
                            save_path = output_label_dir / f'{sample_name}.npz'
                            
                            np.savez_compressed(
                                save_path,
                                frames=frames_array,
                                hand_landmarks=hand_lms_array,
                                pose_landmarks=pose_lms_array,
                                label=label
                            )
                            total_samples += 1
                            class_samples += 1
                    
                    except Exception as e:
                        print(f"Error processing {video_file}: {e}")
            
            # Process extracted video frames from subfolders
            if use_images:  # Changed from use_videos to use_images since these are image frames
                for frames_dir in frames_dirs:
                    try:
                        frame_files = sorted([f for f in frames_dir.iterdir() 
                                            if f.suffix.lower() in image_extensions])
                        
                        if len(frame_files) > 0:
                            frames, hand_lms, pose_lms = self.process_images(frame_files)
                            
                            if len(frames) > 0:
                                frames_array = np.array(frames, dtype=np.uint8)
                                hand_lms_array = np.array(hand_lms, dtype=np.float32)
                                pose_lms_array = np.array(pose_lms, dtype=np.float32)
                                
                                # Save with descriptive name
                                sample_name = f'{frames_dir.name}_{label}'
                                save_path = output_label_dir / f'{sample_name}.npz'
                                
                                np.savez_compressed(
                                    save_path,
                                    frames=frames_array,
                                    hand_landmarks=hand_lms_array,
                                    pose_landmarks=pose_lms_array,
                                    label=label
                                )
                                total_samples += 1
                                class_samples += 1
                    
                    except Exception as e:
                        print(f"Error processing frames in {frames_dir}: {e}")
            
            # Process images (for static dataset - 26,000 images)
            if use_images and images:
                # Sort images
                images = sorted(images)
                
                # For static datasets with many images per class
                # Create multiple samples by grouping images
                images_per_sample = self.frames_per_clip
                
                for i in range(0, len(images), max(1, len(images) // 50)):  # Create ~50 samples per class
                    # Select images for this sample
                    start_idx = i
                    end_idx = min(i + images_per_sample, len(images))
                    
                    if end_idx - start_idx < 3:  # Skip if too few images
                        continue
                    
                    image_group = images[start_idx:end_idx]
                    
                    try:
                        frames, hand_lms, pose_lms = self.process_images(image_group)
                        
                        if len(frames) > 0:
                            frames_array = np.array(frames, dtype=np.uint8)
                            hand_lms_array = np.array(hand_lms, dtype=np.float32)
                            pose_lms_array = np.array(pose_lms, dtype=np.float32)
                            
                            # Save
                            sample_name = f'sample_{i:05d}'
                            save_path = output_label_dir / f'{sample_name}.npz'
                            
                            np.savez_compressed(
                                save_path,
                                frames=frames_array,
                                hand_landmarks=hand_lms_array,
                                pose_landmarks=pose_lms_array,
                                label=label
                            )
                            total_samples += 1
                            class_samples += 1
                    
                    except Exception as e:
                        print(f"Error processing images {start_idx}-{end_idx}: {e}")
            
            class_stats[label] = class_samples
            if class_samples > 0:
                print(f"  {label}: {class_samples} samples")
        
        print("\nConversion complete!")
        print(f"Total samples created: {total_samples}")
        print(f"Output directory: {output_path}")
        
        # Create label mapping
        labels = sorted([d.name for d in class_dirs])
        label_mapping = {label: idx for idx, label in enumerate(labels)}
        
        label_map_path = Path(output_dir) / 'label_mapping.json'
        with open(label_map_path, 'w') as f:
            json.dump(label_mapping, f, indent=2)
        
        print(f"Label mapping saved to: {label_map_path}")
        print(f"Labels: {labels}")
        
        # Update config with number of classes
        self.config['model']['num_classes'] = len(labels)
        with open('configs/config.yaml', 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        
        print(f"Updated config.yaml with {len(labels)} classes")
    
    def __del__(self):
        self.hands.close()
        self.pose.close()


def main():
    parser = argparse.ArgumentParser(description='Convert A-Z + gestures sign language dataset')
    parser.add_argument('--input', type=str, required=True, help='Input directory with class folders')
    parser.add_argument('--output', type=str, default='data', help='Output directory')
    parser.add_argument('--split', type=str, default='labeled', choices=['labeled', 'unlabeled'], 
                       help='Dataset split type')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Config file')
    parser.add_argument('--no-videos', action='store_true', help='Skip processing videos')
    parser.add_argument('--no-images', action='store_true', help='Skip processing static images')
    
    args = parser.parse_args()
    
    use_videos = not args.no_videos
    use_images = not args.no_images
    
    converter = AZDatasetConverter(args.config)
    converter.convert_dataset(args.input, args.output, args.split, use_videos, use_images)
    
    print("\nNext steps:")
    print("1. Run: python src/preprocess.py")
    print("2. Run: python create_models.py")
    print("3. Run: python src/train.py")


if __name__ == '__main__':
    main()
