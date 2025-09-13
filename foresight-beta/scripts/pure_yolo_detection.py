#!/usr/bin/env python3
"""
Pure YOLO Detection - Optimized with ChatGPT's anti-flicker recommendations
No fallback detection, just pure YOLO with tracking and FPS limiting
"""

import cv2
import numpy as np
from mss import mss
import time
import os
from ultralytics import YOLO

# ChatGPT recommendation #1: Move model initialization to module level
print("[INFO] Initializing YOLO model at module level...")
os.environ['YOLO_VERBOSE'] = 'False'
GLOBAL_MODEL = None

try:
    import os
    # Disable PyTorch weights_only restriction
    os.environ['PYTORCH_DISABLE_WEIGHTS_ONLY_LOAD'] = '1'
    
    import torch
    from ultralytics import YOLO
    
    GLOBAL_MODEL = YOLO('yolov8n.pt')
    print("[INFO] Global YOLO model initialized successfully")
except Exception as e:
    print(f"[ERROR] Failed to initialize global YOLO model: {e}")
    GLOBAL_MODEL = None

class PureYOLODetector:
    def __init__(self, model_name='yolov8n.pt', confidence_threshold=0.4):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        
        # ChatGPT recommendation #1: Use global model instance
        self.model = GLOBAL_MODEL
        
        # ULTRA-HIGH FREQUENCY: Maximum detection speed - no FPS limits
        self.last_inference_time = 0
        self.detection_count = 0
        
        # Check for CUDA-enabled GPU (like reference code)
        self.device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
        print(f"[INFO] Using device: {self.device}")
        
        if self.model and self.device == "cuda":
            self.model.to(self.device)
            print("[INFO] Model moved to GPU")
    
    def initialize_model(self):
        """Initialize YOLO model exactly like reference code"""
        try:
            # Set environment variables to reduce verbosity
            os.environ['YOLO_VERBOSE'] = 'False'
            
            # Import YOLO
            from ultralytics import YOLO
            
            print(f"[INFO] Loading YOLO model: {self.model_name}")
            
            # Load the YOLO model (without 'device' argument in the constructor)
            self.model = YOLO(self.model_name)
            
            # Move model to GPU if available (like reference code)
            if self.device == "cuda":
                self.model.to(self.device)
                print("[INFO] Model moved to GPU")
            
            print("[INFO] YOLO model initialized successfully")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize YOLO model: {e}")
            self.model = None
            return False
    
    def detect_and_draw(self, frame):
        """Optimized detect and draw with ChatGPT's anti-flicker recommendations"""
        if self.model is None:
            return frame
        
        try:
            # ULTRA-HIGH FREQUENCY: No FPS limiting - detect every millisecond
            current_time = time.time()
            self.detection_count += 1
            
            # ChatGPT recommendation #5: Use model.track() with persistence
            try:
                results = self.model.track(
                    frame,
                    persist=True,
                    tracker='bytetrack.yaml',
                    conf=self.confidence_threshold,
                    iou=0.6,  # ChatGPT recommendation #8: Higher IoU to reduce false positives
                    verbose=False
                )
            except:
                # Fallback to regular detection if tracking fails
                results = self.model(frame, conf=self.confidence_threshold, verbose=False, show=False)
            
            # Draw boxes on the frame - IMMEDIATE RENDERING
            annotated = results[0].plot()
            
            # Update timestamp for performance tracking
            self.last_inference_time = current_time
            
            return annotated
            
        except Exception as e:
            print(f"[ERROR] Detection failed: {e}")
            return frame
    
    def get_detections(self, frame):
        """Get raw detection data"""
        if self.model is None:
            return []
        
        try:
            # Run YOLO model inference
            results = self.model(frame, conf=self.confidence_threshold, verbose=False, show=False)
            
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
                        
                        # Get class name from YOLO's built-in names
                        class_name = self.model.names[class_id] if class_id in self.model.names else f'class_{class_id}'
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'label': class_name
                        }
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"[ERROR] Detection failed: {e}")
            return []

# Test the pure YOLO detector
if __name__ == "__main__":
    detector = PureYOLODetector(confidence_threshold=0.4)  # ChatGPT recommendation #8: Balanced confidence
    
    if detector.model is None:
        print("[ERROR] YOLO model failed to initialize")
        exit(1)
    
    sct = mss()
    
    # Screen resolution (adjust based on your setup)
    screen_width = 1920
    screen_height = 1080
    
    # Capture size (reduced resolution for performance)
    capture_width = 640
    capture_height = 480
    
    # Monitor area using final YOLO coordinates
    monitor = {
        "top": 119,
        "left": 139,
        "width": 1498,
        "height": 936
    }
    
    print(f"[INFO] Capturing lower-right area: {monitor}")
    
    # Create YOLO output window (resizable, like reference)
    cv2.namedWindow("Pure YOLO Detection", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Pure YOLO Detection", 50, 50)
    cv2.resizeWindow("Pure YOLO Detection", 640, 480)
    
    # FPS tracking
    fps = 0
    prev_time = time.time()
    
    print("[INFO] Starting ULTRA-HIGH FREQUENCY YOLO detection - MILLISECOND LEVEL...")
    print("[INFO] Press 'q' to quit")
    
    # MAXIMUM SPEED: No delays, no FPS calculations - pure detection speed
    detection_counter = 0
    start_time = time.time()
    
    while True:
        # Capture frame from screen - IMMEDIATE
        frame = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        # Run ULTRA-HIGH FREQUENCY YOLO detection - NO LIMITS
        annotated_frame = detector.detect_and_draw(frame)
        
        # Show the result in the window - IMMEDIATE
        cv2.imshow("Pure YOLO Detection - MILLISECOND MODE", annotated_frame)
        
        # Track detection frequency
        detection_counter += 1
        if detection_counter % 1000 == 0:  # Every 1000 detections
            elapsed = time.time() - start_time
            print(f"[ULTRA-SPEED] {detection_counter} detections in {elapsed:.2f}s = {detection_counter/elapsed:.1f} det/sec")
        
        # IMMEDIATE EXIT CHECK - no delays
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print("[INFO] Pure YOLO detection test completed")