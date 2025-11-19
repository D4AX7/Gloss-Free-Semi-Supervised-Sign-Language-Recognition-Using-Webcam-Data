import cv2
import mediapipe as mp
import numpy as np
import os
import argparse
import yaml
from pathlib import Path
import json
from datetime import datetime


class DataCollector:
    def __init__(self, config_path='configs/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
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
        
        # Pad to always have 2 hands (even if empty)
        while len(hand_landmarks) < 2:
            hand_landmarks.append([0.0] * 63)  # 21 landmarks * 3 coords
        
        # Process pose
        pose_results = self.pose.process(image_rgb)
        pose_landmarks = []
        
        if pose_results.pose_landmarks:
            for lm in pose_results.pose_landmarks.landmark:
                pose_landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            pose_landmarks = [0.0] * 132  # 33 landmarks * 4 values
        
        return {
            'hand_landmarks': np.array(hand_landmarks, dtype=np.float32),
            'pose_landmarks': np.array(pose_landmarks, dtype=np.float32)
        }
    
    def collect_video(self, label, num_samples, output_dir, is_labeled=True):
        """Collect video samples with webcam"""
        data_type = 'labeled' if is_labeled else 'unlabeled'
        save_dir = Path(output_dir) / data_type / label
        save_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        fps = self.config['data']['fps']
        frames_per_clip = self.config['data']['frames_per_clip']
        
        print(f"Collecting {num_samples} samples for label '{label}'")
        print("Press SPACE to start recording, 'q' to quit")
        
        sample_idx = len(list(save_dir.glob('*.npz')))
        
        while sample_idx < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            cv2.putText(frame, f"Sample {sample_idx + 1}/{num_samples}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press SPACE to record", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                # Record clip
                frames = []
                landmarks_list = []
                
                print(f"Recording sample {sample_idx + 1}...")
                
                for i in range(frames_per_clip):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Resize frame
                    h, w = self.config['data']['frame_height'], self.config['data']['frame_width']
                    frame_resized = cv2.resize(frame, (w, h))
                    frames.append(frame_resized)
                    
                    # Extract landmarks
                    landmarks = self.extract_landmarks(frame)
                    landmarks_list.append(landmarks)
                    
                    # Display
                    cv2.putText(frame, f"Recording: {i + 1}/{frames_per_clip}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow('Data Collection', frame)
                    cv2.waitKey(1)
                
                if len(frames) == frames_per_clip:
                    # Save data
                    frames_array = np.array(frames, dtype=np.uint8)
                    hand_lms = np.array([lm['hand_landmarks'] for lm in landmarks_list], dtype=np.float32)
                    pose_lms = np.array([lm['pose_landmarks'] for lm in landmarks_list], dtype=np.float32)
                    
                    save_path = save_dir / f'sample_{sample_idx:04d}.npz'
                    np.savez_compressed(
                        save_path,
                        frames=frames_array,
                        hand_landmarks=hand_lms,
                        pose_landmarks=pose_lms,
                        label=label,
                        timestamp=datetime.now().isoformat()
                    )
                    
                    print(f"Saved: {save_path}")
                    sample_idx += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"Collection complete. Saved {sample_idx} samples to {save_dir}")
    
    def __del__(self):
        self.hands.close()
        self.pose.close()


def main():
    parser = argparse.ArgumentParser(description='Collect sign language data from webcam')
    parser.add_argument('--label', type=str, required=True, help='Label for the sign')
    parser.add_argument('--samples', type=int, default=50, help='Number of samples to collect')
    parser.add_argument('--output', type=str, default='data', help='Output directory')
    parser.add_argument('--unlabeled', action='store_true', help='Collect as unlabeled data')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    collector = DataCollector(args.config)
    collector.collect_video(
        label=args.label,
        num_samples=args.samples,
        output_dir=args.output,
        is_labeled=not args.unlabeled
    )


if __name__ == '__main__':
    main()
