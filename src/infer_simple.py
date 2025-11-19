import cv2
import torch
import yaml
import numpy as np
import mediapipe as mp
from pathlib import Path
import json
import time

from model import create_model
from utils import load_checkpoint


class SimpleInference:
    def __init__(self):
        # Load config
        with open('configs/config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        print("Loading model...")
        # Load model
        self.model = create_model(self.config)
        load_checkpoint('models/checkpoint.pth', self.model)  # Using 6-epoch trained model
        self.model.eval()
        
        # Load labels
        with open('data/label_mapping.json', 'r') as f:
            label_mapping = json.load(f)
            self.idx_to_label = {v: k for k, v in label_mapping.items()}
        
        print(f"Loaded {len(self.idx_to_label)} labels: {list(self.idx_to_label.values())}")
        
        # MediaPipe - simplest config
        mp_hands = mp.solutions.hands
        mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands_module = mp_hands
        
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.5
        )
        
        # Buffers
        self.frames = []
        self.hand_landmarks_list = []
        self.pose_landmarks_list = []
        self.frames_needed = 16
        
        print("System ready!")
    
    def extract_landmarks(self, frame):
        """Extract landmarks"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Hands
        hand_results = self.hands.process(rgb)
        hand_lms = []
        
        if hand_results.multi_hand_landmarks:
            for hand in hand_results.multi_hand_landmarks:
                for lm in hand.landmark:
                    hand_lms.extend([lm.x, lm.y, lm.z])
        
        # Pad to 2 hands
        while len(hand_lms) < 126:
            hand_lms.append(0.0)
        hand_lms = hand_lms[:126]
        
        # Pose
        pose_results = self.pose.process(rgb)
        pose_lms = []
        
        if pose_results.pose_landmarks:
            for lm in pose_results.pose_landmarks.landmark:
                pose_lms.extend([lm.x, lm.y, lm.z, lm.visibility])
        
        if len(pose_lms) < 132:
            pose_lms = [0.0] * 132
        
        return np.array(hand_lms, dtype=np.float32), np.array(pose_lms, dtype=np.float32), hand_results
    
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n" + "="*50)
        print("WEBCAM STARTED - Show ASL signs!")
        print("="*50)
        print("Tips:")
        print("  - Face camera directly")
        print("  - Good lighting helps")
        print("  - Hold sign for 3-4 seconds")
        print("  - Press 'q' to quit")
        print("="*50 + "\n")
        
        prediction = "Collecting frames..."
        confidence = 0.0
        frame_count = 0
        last_prediction_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every frame
            frame_small = cv2.resize(frame, (224, 224))
            frame_normalized = frame_small.astype(np.float32) / 255.0
            
            hand_lms, pose_lms, hand_results = self.extract_landmarks(frame)
            
            # Add to buffer
            self.frames.append(frame_normalized)
            self.hand_landmarks_list.append(hand_lms)
            self.pose_landmarks_list.append(pose_lms)
            
            # Keep only last 16 frames
            if len(self.frames) > self.frames_needed:
                self.frames.pop(0)
                self.hand_landmarks_list.pop(0)
                self.pose_landmarks_list.pop(0)
            
            # Predict every 2 seconds
            if len(self.frames) == self.frames_needed and time.time() - last_prediction_time > 2.0:
                try:
                    # Prepare data
                    frames_array = np.array(self.frames)  # (16, 224, 224, 3)
                    hand_array = np.array(self.hand_landmarks_list)  # (16, 126)
                    pose_array = np.array(self.pose_landmarks_list)  # (16, 132)
                    
                    # To tensors
                    frames_tensor = torch.from_numpy(frames_array).float()
                    frames_tensor = frames_tensor.permute(3, 0, 1, 2).unsqueeze(0)  # (1, 3, 16, 224, 224)
                    
                    hand_tensor = torch.from_numpy(hand_array).float().unsqueeze(0)
                    pose_tensor = torch.from_numpy(pose_array).float().unsqueeze(0)
                    
                    # Predict
                    with torch.no_grad():
                        logits = self.model(frames_tensor, hand_tensor, pose_tensor)
                        probs = torch.softmax(logits, dim=1)
                        conf, pred = torch.max(probs, dim=1)
                        
                        # Get top 3 predictions for debugging
                        top_probs, top_indices = torch.topk(probs, 3, dim=1)
                    
                    pred_idx = pred.item()
                    confidence = conf.item()
                    
                    # Show top 3 predictions
                    print(f"\n--- Prediction Update ---")
                    print(f"Top 3 predictions:")
                    for i in range(3):
                        idx = top_indices[0][i].item()
                        prob = top_probs[0][i].item()
                        label = self.idx_to_label.get(idx, "Unknown")
                        print(f"  {i+1}. {label}: {prob:.1%}")
                    
                    # LOWER threshold to 0.3 to see ANY predictions
                    if confidence > 0.3:
                        prediction = self.idx_to_label.get(pred_idx, "Unknown")
                        print(f">>> SHOWING: {prediction} ({confidence:.1%})")
                    else:
                        print(f">>> Confidence too low ({confidence:.1%}), not showing")
                    
                    last_prediction_time = time.time()
                    
                except Exception as e:
                    print(f"Error in prediction: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Display
            display = frame.copy()
            
            # Draw hand landmarks
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        display,
                        hand_landmarks,
                        self.mp_hands_module.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                        self.mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2)
                    )
            
            # Text background
            cv2.rectangle(display, (5, 5), (635, 120), (0, 0, 0), -1)
            
            # Status
            buffer_status = f"Buffer: {len(self.frames)}/{self.frames_needed}"
            cv2.putText(display, buffer_status, (15, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # Prediction
            pred_text = f"Sign: {prediction}"
            cv2.putText(display, pred_text, (15, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            conf_text = f"Confidence: {confidence:.1%}"
            cv2.putText(display, conf_text, (15, 105), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Instructions
            cv2.putText(display, "Hold sign for 3 seconds | Press 'q' to quit", 
                       (15, display.shape[0] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.imshow('ASL Recognition - Simple Version', display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        self.pose.close()
        print("\nStopped!")


if __name__ == '__main__':
    try:
        inference = SimpleInference()
        inference.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
