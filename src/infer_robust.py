"""
Robust Real-time ASL Recognition with Improved Preprocessing
Matches training data preprocessing more closely
"""

import cv2
import numpy as np
import torch
import mediapipe as mp
import time
import json
from pathlib import Path
from collections import deque

from model import SignLanguageRecognitionModel


class RobustASLRecognizer:
    def __init__(self, model_path='models/best_model.pth', labels_path='data/label_mapping.json', config_path='configs/config.yaml'):
        print("Loading model...")
        self.device = torch.device('cpu')
        
        # Load config
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load model
        self.model = SignLanguageRecognitionModel(config)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Loaded checkpoint from {model_path}")
        
        # Load labels
        import json
        with open(labels_path, 'r') as f:
            label_mapping = json.load(f)
            self.idx_to_label = {v: k for k, v in label_mapping.items()}
        print(f"Loaded {len(self.idx_to_label)} labels: {list(self.idx_to_label.values())}")
        
        # MediaPipe with optimized settings
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,  # Higher complexity for better accuracy
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Buffers with smoothing
        self.frames_needed = 16
        self.frames_buffer = deque(maxlen=self.frames_needed)
        self.hand_buffer = deque(maxlen=self.frames_needed)
        self.pose_buffer = deque(maxlen=self.frames_needed)
        
        # Prediction smoothing
        self.prediction_history = deque(maxlen=5)
        
        print("System ready!")
    
    def preprocess_frame(self, frame):
        """
        Preprocess frame to match training data EXACTLY:
        - Just resize to 224x224
        - Keep as uint8 [0-255] - NO normalization
        Training data is stored as raw uint8 values!
        """
        # Resize
        frame = cv2.resize(frame, (224, 224))
        
        # Keep as uint8 - training data is NOT normalized!
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
        
        return np.array(hand_lms, dtype=np.float32), np.array(pose_lms, dtype=np.float32), hand_results
    
    def smooth_prediction(self, pred_idx, confidence):
        """Smooth predictions using voting"""
        self.prediction_history.append(pred_idx)
        
        if len(self.prediction_history) < 3:
            return pred_idx, confidence
        
        # Majority voting
        from collections import Counter
        counts = Counter(self.prediction_history)
        most_common = counts.most_common(1)[0]
        
        if most_common[1] >= 2:  # At least 2 same predictions
            return most_common[0], confidence
        
        return pred_idx, confidence
    
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\n" + "="*60)
        print("  ROBUST ASL RECOGNITION - WEBCAM MODE")
        print("="*60)
        print("Instructions:")
        print("  1. Position hand clearly in center of frame")
        print("  2. Ensure good lighting (face a window/light)")
        print("  3. Keep hand steady for 2-3 seconds")
        print("  4. Try different signs from A-Z, HELLO, YES, NO, etc.")
        print("  5. Press 'q' to quit")
        print("="*60 + "\n")
        
        prediction = "Ready..."
        confidence = 0.0
        last_prediction_time = time.time()
        fps_start = time.time()
        frame_counter = 0
        fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # FPS calculation
            frame_counter += 1
            if time.time() - fps_start > 1.0:
                fps = frame_counter
                frame_counter = 0
                fps_start = time.time()
            
            # Mirror for natural interaction
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            
            # Extract landmarks first (from original frame)
            hand_lms, pose_lms, hand_results = self.extract_landmarks(frame)
            
            # Preprocess for model
            processed_frame = self.preprocess_frame(frame)
            
            # Add to buffers
            self.frames_buffer.append(processed_frame)
            self.hand_buffer.append(hand_lms)
            self.pose_buffer.append(pose_lms)
            
            # Draw hand skeleton
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        display_frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )
            
            # Predict when buffer is full (every 1.5 seconds)
            if len(self.frames_buffer) == self.frames_needed and \
               time.time() - last_prediction_time > 1.5:
                try:
                    # Prepare tensors - MUST match training exactly!
                    frames_array = np.array(list(self.frames_buffer))  # (16, 224, 224, 3) uint8 [0-255]
                    hand_array = np.array(list(self.hand_buffer))      # (16, 126)
                    pose_array = np.array(list(self.pose_buffer))      # (16, 132)
                    
                    # Transpose for PyTorch (N, C, T, H, W) and convert to float32
                    # Keep values in 0-255 range - same as training!
                    frames_tensor = torch.from_numpy(frames_array).float().permute(3, 0, 1, 2).unsqueeze(0)
                    hand_tensor = torch.from_numpy(hand_array).unsqueeze(0)
                    pose_tensor = torch.from_numpy(pose_array).unsqueeze(0)
                    
                    # Predict
                    with torch.no_grad():
                        logits = self.model(frames_tensor, hand_tensor, pose_tensor)
                        probs = torch.softmax(logits, dim=1)
                        conf, pred = torch.max(probs, dim=1)
                        
                        pred_idx = pred.item()
                        confidence_raw = conf.item()
                        
                        # Get top 5 for debugging
                        top_probs, top_indices = torch.topk(probs, 5, dim=1)
                        
                        print(f"\n{'='*50}")
                        print(f"Top 5 predictions:")
                        for i in range(5):
                            idx = top_indices[0][i].item()
                            prob = top_probs[0][i].item()
                            label = self.idx_to_label.get(idx, "Unknown")
                            print(f"  {i+1}. {label:12s} : {prob:6.1%}")
                        
                        # Apply smoothing
                        pred_idx, confidence = self.smooth_prediction(pred_idx, confidence_raw)
                        
                        # Update display
                        prediction = self.idx_to_label.get(pred_idx, "Unknown")
                        
                        print(f"\n>>> FINAL: {prediction} ({confidence:.1%})")
                        print(f"{'='*50}")
                        
                    last_prediction_time = time.time()
                
                except Exception as e:
                    print(f"ERROR during prediction: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Display info
            buffer_status = f"Buffer: {len(self.frames_buffer)}/{self.frames_needed}"
            
            # Background for text
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (10, 10), (500, 140), (0, 0, 0), -1)
            display_frame = cv2.addWeighted(display_frame, 0.7, overlay, 0.3, 0)
            
            # Text
            cv2.putText(display_frame, "ROBUST ASL RECOGNITION", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_frame, buffer_status, (20, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Prediction with confidence color coding
            if confidence > 0.7:
                color = (0, 255, 0)  # Green - high confidence
            elif confidence > 0.5:
                color = (0, 255, 255)  # Yellow - medium
            else:
                color = (0, 165, 255)  # Orange - low
            
            cv2.putText(display_frame, f"Sign: {prediction}", (20, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(display_frame, f"Confidence: {confidence:.1%}", (20, 125),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # FPS
            cv2.putText(display_frame, f"FPS: {fps}", (display_frame.shape[1] - 100, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('ASL Recognition', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\nStopped!")


if __name__ == "__main__":
    recognizer = RobustASLRecognizer()
    recognizer.run()
