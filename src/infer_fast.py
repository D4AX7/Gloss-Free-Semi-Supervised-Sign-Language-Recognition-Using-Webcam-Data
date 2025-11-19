import cv2
import torch
import yaml
import argparse
from pathlib import Path
import numpy as np
import mediapipe as mp
from collections import deque
import json
import time

from model import create_model
from utils import load_checkpoint


class FastRealTimeInference:
    def __init__(self, config_path, model_path, label_mapping_path, device='cpu'):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = 'cpu'  # Force CPU for stability
        
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
        
        # MediaPipe - OPTIMIZED for speed
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
            model_complexity=0,  # Fastest model
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        
        # Sliding window
        self.frames_per_clip = self.config['data']['frames_per_clip']
        self.stride = 8  # Process every 8th frame for speed
        self.frame_buffer = deque(maxlen=self.frames_per_clip)
        self.hand_buffer = deque(maxlen=self.frames_per_clip)
        self.pose_buffer = deque(maxlen=self.frames_per_clip)
        
        # Prediction smoothing
        self.smoothing_window = 5
        self.prediction_buffer = deque(maxlen=self.smoothing_window)
        
        # Confidence threshold
        self.confidence_threshold = 0.5
        
        # Frame size - REDUCED for speed
        self.frame_height = 224
        self.frame_width = 224
        
        # Skip frames
        self.skip_frames = 2  # Process every 2nd frame
        
        print(f"Fast Model loaded. Using device: {self.device}")
        print(f"Labels: {list(self.idx_to_label.values())}")
    
    def extract_landmarks(self, frame):
        """Extract hand and pose landmarks - OPTIMIZED"""
        # Resize for MediaPipe (smaller = faster)
        small_frame = cv2.resize(frame, (320, 240))
        image_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
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
        
        return hand_landmarks, pose_landmarks, hand_results, pose_results
    
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
        """Run real-time inference - OPTIMIZED"""
        cap = cv2.VideoCapture(video_source)
        
        # Set camera properties for speed
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source {video_source}")
            return
        
        print("Starting FAST real-time inference. Press 'q' to quit.")
        print("Tips: Good lighting and clear background improve accuracy!")
        print("Hold sign steady for 2-3 seconds for prediction.")
        
        frame_count = 0
        process_count = 0
        current_prediction = "Waiting..."
        current_confidence = 0.0
        
        # FPS tracking
        fps_start_time = time.time()
        fps_counter = 0
        current_fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames for speed
            if frame_count % self.skip_frames != 0:
                frame_count += 1
                
                # Still show display even when skipping
                display_frame = frame.copy()
                
                # Draw prediction
                text = f"Sign: {current_prediction}"
                conf_text = f"Confidence: {current_confidence:.1%}"
                fps_text = f"FPS: {current_fps:.1f}"
                
                cv2.putText(display_frame, text, (10, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.putText(display_frame, conf_text, (10, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(display_frame, fps_text, (10, 125),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 255), 2)
                cv2.putText(display_frame, "Press 'q' to quit", (10, display_frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                
                cv2.imshow('ASL Sign Recognition', display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            # Preprocess frame
            processed_frame = self.preprocess_frame(frame)
            
            # Extract landmarks (with visualization data)
            hand_lms, pose_lms, hand_results, pose_results = self.extract_landmarks(frame)
            
            # Add to buffer
            self.frame_buffer.append(processed_frame)
            self.hand_buffer.append(hand_lms)
            self.pose_buffer.append(pose_lms)
            
            # Run prediction every stride frames
            if process_count % self.stride == 0 and len(self.frame_buffer) == self.frames_per_clip:
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
            
            # Draw hand landmarks
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        display_frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                    )
            
            # Draw prediction
            text = f"Sign: {current_prediction}"
            conf_text = f"Confidence: {current_confidence:.1%}"
            
            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_start_time > 1.0:
                current_fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
            
            fps_text = f"FPS: {current_fps:.1f}"
            
            # Background rectangles for text
            cv2.rectangle(display_frame, (5, 10), (500, 140), (0, 0, 0), -1)
            
            cv2.putText(display_frame, text, (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(display_frame, conf_text, (10, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(display_frame, fps_text, (10, 125),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 255), 2)
            cv2.putText(display_frame, "Press 'q' to quit", (10, display_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            
            # Show frame
            cv2.imshow('ASL Sign Recognition', display_frame)
            
            frame_count += 1
            process_count += 1
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Cleanup
        self.hands.close()
        self.pose.close()
        
        print("\nInference stopped. Thank you!")


def main():
    parser = argparse.ArgumentParser(description='FAST Real-time sign language recognition')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Config file')
    parser.add_argument('--model', type=str, default='models/best_model.pth', help='Trained model path')
    parser.add_argument('--label_mapping', type=str, default='data/label_mapping.json', help='Label mapping path')
    parser.add_argument('--video', type=str, default='0', help='Video source (0 for webcam or path to video file)')
    
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
    inference = FastRealTimeInference(
        args.config,
        args.model,
        args.label_mapping,
        device='cpu'
    )
    
    # Run
    inference.run(video_source)


if __name__ == '__main__':
    main()
