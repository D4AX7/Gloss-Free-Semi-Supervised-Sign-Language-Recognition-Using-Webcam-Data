import cv2
import torch
import yaml
import argparse
from pathlib import Path
import numpy as np
import mediapipe as mp
from collections import deque
import json

from model import create_model
from utils import load_checkpoint


class RealTimeInference:
    def __init__(self, config_path, model_path, label_mapping_path, device='cuda'):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Load model
        self.model = create_model(self.config).to(self.device)
        load_checkpoint(model_path, self.model)
        self.model.eval()
        
        # Load label mapping
        self.idx_to_label = {}
        if Path(label_mapping_path).exists():
            with open(label_mapping_path, 'r') as f:
                label_mapping = json.load(f)
                self.idx_to_label = {v: k for k, v in label_mapping.items()}
        
        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # Reduced complexity for speed
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        
        # Sliding window
        self.frames_per_clip = self.config['data']['frames_per_clip']
        self.stride = self.config['inference']['sliding_window_stride']
        self.frame_buffer = deque(maxlen=self.frames_per_clip)
        self.hand_buffer = deque(maxlen=self.frames_per_clip)
        self.pose_buffer = deque(maxlen=self.frames_per_clip)
        
        # Prediction smoothing
        self.smoothing_window = self.config['inference']['smoothing_window']
        self.prediction_buffer = deque(maxlen=self.smoothing_window)
        
        # Confidence threshold
        self.confidence_threshold = self.config['inference']['confidence_threshold']
        
        # Frame size
        self.frame_height = self.config['data']['frame_height']
        self.frame_width = self.config['data']['frame_width']
        
        print(f"Model loaded. Using device: {self.device}")
        print(f"Labels: {list(self.idx_to_label.values())}")
    
    def extract_landmarks(self, frame):
        """Extract hand and pose landmarks"""
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
        
        hand_landmarks = np.array(hand_landmarks, dtype=np.float32).flatten()
        
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
    
    def preprocess_frame(self, frame):
        """Preprocess single frame"""
        # Resize
        frame_resized = cv2.resize(frame, (self.frame_width, self.frame_height))
        # Normalize
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        return frame_normalized
    
    def predict(self):
        """Run prediction on current buffer"""
        if len(self.frame_buffer) < self.frames_per_clip:
            return None, 0.0
        
        # Stack frames
        frames = np.array(list(self.frame_buffer))  # (T, H, W, C)
        hand_lms = np.array(list(self.hand_buffer))  # (T, 126)
        pose_lms = np.array(list(self.pose_buffer))  # (T, 132)
        
        # Convert to tensors
        frames = torch.from_numpy(frames).float()
        hand_lms = torch.from_numpy(hand_lms).float()
        pose_lms = torch.from_numpy(pose_lms).float()
        
        # Permute frames to (C, T, H, W)
        frames = frames.permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, T, H, W)
        hand_lms = hand_lms.unsqueeze(0)  # (1, T, 126)
        pose_lms = pose_lms.unsqueeze(0)  # (1, T, 132)
        
        # Move to device
        frames = frames.to(self.device)
        hand_lms = hand_lms.to(self.device)
        pose_lms = pose_lms.to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(frames, hand_lms, pose_lms)
            probs = torch.softmax(logits, dim=1)
            confidence, prediction = torch.max(probs, dim=1)
        
        pred_idx = prediction.item()
        conf = confidence.item()
        
        return pred_idx, conf
    
    def smooth_predictions(self, pred_idx):
        """Smooth predictions using majority voting"""
        self.prediction_buffer.append(pred_idx)
        
        if len(self.prediction_buffer) < self.smoothing_window:
            return pred_idx
        
        # Majority voting
        counts = {}
        for p in self.prediction_buffer:
            counts[p] = counts.get(p, 0) + 1
        
        smoothed_pred = max(counts, key=counts.get)
        return smoothed_pred
    
    def run(self, video_source=0):
        """Run real-time inference"""
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source {video_source}")
            return
        
        print("Starting real-time inference. Press 'q' to quit.")
        
        frame_count = 0
        current_prediction = "Waiting..."
        current_confidence = 0.0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess frame
            processed_frame = self.preprocess_frame(frame)
            
            # Extract landmarks
            hand_lms, pose_lms = self.extract_landmarks(frame)
            
            # Add to buffer
            self.frame_buffer.append(processed_frame)
            self.hand_buffer.append(hand_lms)
            self.pose_buffer.append(pose_lms)
            
            # Run prediction every stride frames
            if frame_count % self.stride == 0 and len(self.frame_buffer) == self.frames_per_clip:
                pred_idx, confidence = self.predict()
                
                if pred_idx is not None and confidence >= self.confidence_threshold:
                    # Smooth prediction
                    smoothed_idx = self.smooth_predictions(pred_idx)
                    
                    # Get label
                    if smoothed_idx in self.idx_to_label:
                        current_prediction = self.idx_to_label[smoothed_idx]
                        current_confidence = confidence
            
            # Display
            display_frame = frame.copy()
            
            # Draw prediction
            text = f"Sign: {current_prediction}"
            conf_text = f"Confidence: {current_confidence:.2%}"
            
            cv2.putText(display_frame, text, (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.putText(display_frame, conf_text, (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Draw FPS
            fps_text = f"Frame: {frame_count}"
            cv2.putText(display_frame, fps_text, (10, display_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow('Sign Language Recognition', display_frame)
            
            frame_count += 1
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Cleanup
        self.hands.close()
        self.pose.close()


def main():
    parser = argparse.ArgumentParser(description='Real-time sign language recognition')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Config file')
    parser.add_argument('--model', type=str, default='models/best_model.pth', help='Trained model path')
    parser.add_argument('--label_mapping', type=str, default='data/label_mapping.json', help='Label mapping path')
    parser.add_argument('--video', type=str, default='0', help='Video source (0 for webcam or path to video file)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        print("Please train a model first using train.py")
        return
    
    # Parse video source
    try:
        video_source = int(args.video)
    except ValueError:
        video_source = args.video
    
    # Create inference engine
    inference = RealTimeInference(
        args.config,
        args.model,
        args.label_mapping,
        device=args.device
    )
    
    # Run
    inference.run(video_source)


if __name__ == '__main__':
    main()
