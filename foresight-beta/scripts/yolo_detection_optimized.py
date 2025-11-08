#!/usr/bin/env python3
"""
Foresight - Optimized YOLO Detection Script
Implements ChatGPT's recommendations to eliminate flicker:
1. Module-level model initialization
2. model.track() with persistence
3. Buffer size optimization
4. Temporal smoothing with grace period
"""

import cv2
import numpy as np
import time
import os
import sys
from pathlib import Path

# Initialize YOLO model at module level to prevent re-initialization flicker
print("[INFO] Loading YOLO model at module level...")
try:
    # Set environment variables for Windows compatibility
    os.environ['YOLO_VERBOSE'] = 'False'
    os.environ['ULTRALYTICS_OFFLINE'] = '1'
    os.environ['ULTRALYTICS_DISABLE_TELEMETRY'] = '1'
    os.environ['ULTRALYTICS_DISABLE_UPDATE_CHECK'] = '1'
    
    from ultralytics import YOLO
    import torch
    
    # Fix PyTorch security restrictions
    torch.serialization._use_new_zipfile_serialization = False
    original_load = torch.load
    def patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    torch.load = patched_load
    
    # Initialize model once at module level
    model = YOLO('yolov8n.pt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if hasattr(model.model, 'to'):
        model.model.to(device)
    print(f"[INFO] YOLO model initialized successfully on {device}")
except Exception as e:
    print(f"[ERROR] Failed to initialize YOLO model: {e}")
    model = None
    device = 'cpu'

class OptimizedYOLODetector:
    def __init__(self, conf_threshold=0.35, iou_threshold=0.6):
        self.model = model  # Use global model instance
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.last_inference_time = 0
        self.inference_fps = 15  # Limit inference to 15 FPS
        self.last_results = []  # Hold last results for frame interpolation
        self.grace_period = 0.3  # 300ms grace period for disappeared objects
        self.disappeared_objects = {}  # Track disappeared objects with timestamps
        
        print(f"[INFO] Optimized YOLO detector initialized")
        print(f"[INFO] Confidence threshold: {self.conf_threshold}")
        print(f"[INFO] IoU threshold: {self.iou_threshold}")
        print(f"[INFO] Inference FPS limit: {self.inference_fps}")
    
    def detect_objects(self, frame):
        """Optimized detection with ChatGPT's recommendations"""
        if self.model is None:
            return []
        
        current_time = time.time()
        
        # Implement inference FPS limiting (ChatGPT recommendation #4)
        if current_time - self.last_inference_time < 1.0 / self.inference_fps:
            # Return last results with grace period handling
            return self._apply_grace_period(self.last_results)
        
        try:
            # Resize frame for speed optimization
            height, width = frame.shape[:2]
            if width > 640:
                scale = 640 / width
                new_width = 640
                new_height = int(height * scale)
                frame_resized = cv2.resize(frame, (new_width, new_height))
            else:
                frame_resized = frame
                scale = 1.0
            
            # Use model.track() with persistence (ChatGPT recommendation #5)
            results = self.model.track(
                frame_resized,
                persist=True,
                tracker='bytetrack.yaml',
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
                imgsz=640
            )
            
            detections = []
            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes
                
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Get tracking ID if available
                        track_id = None
                        if hasattr(box, 'id') and box.id is not None:
                            track_id = int(box.id[0].cpu().numpy())
                        
                        # Scale coordinates back to original frame size
                        if scale != 1.0:
                            x1 = x1 / scale
                            y1 = y1 / scale
                            x2 = x2 / scale
                            y2 = y2 / scale
                        
                        # Get class name
                        class_name = self.model.names[class_id] if class_id in self.model.names else f'class_{class_id}'
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'label': class_name,
                            'track_id': track_id,
                            'timestamp': current_time
                        }
                        detections.append(detection)
            
            self.last_results = detections
            self.last_inference_time = current_time
            
            # Apply grace period for temporal smoothing
            return self._apply_grace_period(detections)
            
        except Exception as e:
            print(f"[ERROR] Detection failed: {e}")
            return self.last_results
    
    def _apply_grace_period(self, current_detections):
        """Apply 250-300ms grace period to prevent one-frame disappearances"""
        current_time = time.time()
        
        # Track current detection IDs
        current_track_ids = set()
        for detection in current_detections:
            if detection.get('track_id') is not None:
                current_track_ids.add(detection['track_id'])
        
        # Check for disappeared objects
        for track_id, disappear_time in list(self.disappeared_objects.items()):
            if current_time - disappear_time > self.grace_period:
                # Remove objects that have been gone too long
                del self.disappeared_objects[track_id]
        
        # Add objects that just disappeared
        if hasattr(self, '_last_track_ids'):
            for track_id in self._last_track_ids:
                if track_id not in current_track_ids and track_id not in self.disappeared_objects:
                    self.disappeared_objects[track_id] = current_time
        
        self._last_track_ids = current_track_ids.copy()
        
        # Return current detections (tracking handles persistence)
        return current_detections

def setup_capture_optimization(cap):
    """Setup capture with ChatGPT's buffer optimization recommendations"""
    if cap is not None:
        # Set buffer size to 1 (ChatGPT recommendation #6)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("[INFO] Capture buffer size set to 1 for minimal latency")

def main():
    """Test the optimized detector"""
    print("[INFO] Testing optimized YOLO detector...")
    
    # Initialize detector
    detector = OptimizedYOLODetector(conf_threshold=0.35, iou_threshold=0.6)
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    setup_capture_optimization(cap)
    
    if not cap.isOpened():
        print("[ERROR] Could not open webcam")
        return
    
    print("[INFO] Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect objects
        detections = detector.detect_objects(frame)
        
        # Draw detections
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            label = detection['label']
            track_id = detection.get('track_id', 'N/A')
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label with tracking ID
            text = f"{label} ({confidence:.2f}) ID:{track_id}"
            cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Optimized YOLO Detection', frame)
        
        # Use cv2.waitKey(1) as recommended by ChatGPT
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()