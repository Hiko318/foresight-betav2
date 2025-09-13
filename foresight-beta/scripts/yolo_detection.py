#!/usr/bin/env python3
"""
Argus - YOLO Detection Script
This script handles SAR (Search and Rescue) mode detection using YOLO
"""

import cv2
import numpy as np
import argparse
import time
import sys
import os
from pathlib import Path
from collections import defaultdict
try:
    import win32gui
    import win32ui
    import win32con
    import win32api
except ImportError:
    print("[WARNING] pywin32 not available - window overlay features disabled")

class ObjectTracker:
    def __init__(self, smoothing_factor=0.85, max_distance=50, frame_skip_interval=1, grace_period_ms=300):
        self.tracked_objects = {}
        self.next_id = 0
        self.smoothing_factor = smoothing_factor
        self.max_distance = max_distance
        self.frame_skip_counter = 0
        self.frame_skip_interval = frame_skip_interval
        self.previous_detections = []
        self.grace_period_ms = grace_period_ms  # ChatGPT recommendation: 250-300ms grace window
        
    def update(self, detections):
        """Update tracker with new detections"""
        # Frame skipping logic
        self.frame_skip_counter += 1
        if self.frame_skip_counter % self.frame_skip_interval != 0:
            # Return interpolated objects for skipped frames
            return self.get_interpolated_objects()
            
        if not detections:
            return list(self.tracked_objects.values())
            
        # Convert detections to center points for tracking
        new_centers = []
        for detection in detections:
            x, y, w, h = detection['box']
            center_x = x + w / 2
            center_y = y + h / 2
            new_centers.append((center_x, center_y, detection))
        
        # Match detections to existing tracks
        matched_pairs = []
        unmatched_detections = list(range(len(new_centers)))
        unmatched_tracks = list(self.tracked_objects.keys())
        
        # Calculate distances between all tracks and detections
        all_matches = []
        for track_id in self.tracked_objects:
            track = self.tracked_objects[track_id]
            track_center_x = track['box'][0] + track['box'][2] / 2
            track_center_y = track['box'][1] + track['box'][3] / 2
            
            for i, (center_x, center_y, detection) in enumerate(new_centers):
                distance = np.sqrt((track_center_x - center_x)**2 + (track_center_y - center_y)**2)
                if distance < self.max_distance:
                    all_matches.append((distance, track_id, i))
        
        # Sort by distance and assign matches
        all_matches.sort()
        used_tracks = set()
        used_detections = set()
        
        for distance, track_id, detection_idx in all_matches:
            if track_id not in used_tracks and detection_idx not in used_detections:
                matched_pairs.append((track_id, detection_idx))
                used_tracks.add(track_id)
                used_detections.add(detection_idx)
                unmatched_detections.remove(detection_idx)
                unmatched_tracks.remove(track_id)
        
        # Update matched tracks with smoothing
        for track_id, detection_idx in matched_pairs:
            center_x, center_y, detection = new_centers[detection_idx]
            track = self.tracked_objects[track_id]
            
            # Store velocity for prediction
            if 'velocity_x' not in track:
                track['velocity_x'] = 0
                track['velocity_y'] = 0
            else:
                old_center_x = track['box'][0] + track['box'][2] / 2
                old_center_y = track['box'][1] + track['box'][3] / 2
                track['velocity_x'] = center_x - old_center_x
                track['velocity_y'] = center_y - old_center_y
            
            # Enhanced temporal smoothing with adaptive factors
            old_box = track['box']
            new_box = detection['box']
            
            # Adaptive smoothing based on confidence and motion
            confidence_factor = min(detection['confidence'], 0.9)  # Cap at 0.9
            motion_magnitude = abs(track['velocity_x']) + abs(track['velocity_y'])
            motion_factor = min(motion_magnitude / 50.0, 0.5)  # Reduce smoothing for fast motion
            
            # Dynamic smoothing factor: higher confidence = less smoothing, more motion = less smoothing
            dynamic_smoothing = self.smoothing_factor * (1 - confidence_factor * 0.3) * (1 - motion_factor)
            dynamic_smoothing = max(0.3, min(0.9, dynamic_smoothing))  # Clamp between 0.3-0.9
            
            # Apply exponential moving average with dynamic smoothing
            smoothed_box = [
                old_box[0] * dynamic_smoothing + new_box[0] * (1 - dynamic_smoothing),
                old_box[1] * dynamic_smoothing + new_box[1] * (1 - dynamic_smoothing),
                old_box[2] * dynamic_smoothing + new_box[2] * (1 - dynamic_smoothing),
                old_box[3] * dynamic_smoothing + new_box[3] * (1 - dynamic_smoothing)
            ]
            
            # Additional stability: prevent tiny movements (jitter reduction)
            for i in range(4):
                if abs(smoothed_box[i] - old_box[i]) < 2.0:  # Less than 2 pixels
                    smoothed_box[i] = old_box[i]  # Keep old position
            
            track['box'] = smoothed_box
            track['label'] = detection['label']
            track['confidence'] = detection['confidence']
            track['class_id'] = detection['class_id']
            track['last_seen'] = time.time()
        
        # Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            center_x, center_y, detection = new_centers[detection_idx]
            self.tracked_objects[self.next_id] = {
                'box': detection['box'],
                'label': detection['label'],
                'confidence': detection['confidence'],
                'class_id': detection['class_id'],
                'last_seen': time.time(),
                'velocity_x': 0,
                'velocity_y': 0
            }
            self.next_id += 1
        
        # Remove old tracks using grace period (ChatGPT recommendation #5)
        current_time = time.time()
        tracks_to_remove = []
        grace_period_seconds = self.grace_period_ms / 1000.0
        for track_id in unmatched_tracks:
            if current_time - self.tracked_objects[track_id]['last_seen'] > grace_period_seconds:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracked_objects[track_id]
        
        return list(self.tracked_objects.values())
    
    def get_interpolated_objects(self):
        """Get interpolated object positions for skipped frames"""
        interpolated_objects = []
        for track in self.tracked_objects.values():
            # Predict position based on velocity
            predicted_box = track['box'].copy()
            if 'velocity_x' in track and 'velocity_y' in track:
                predicted_box[0] += track['velocity_x']
                predicted_box[1] += track['velocity_y']
            
            interpolated_objects.append({
                'box': predicted_box,
                'label': track['label'],
                'confidence': track['confidence'],
                'class_id': track['class_id']
            })
        
        return interpolated_objects

class StaticDetectionZones:
    def __init__(self, frame_width=640, frame_height=480):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.detection_zones = self._create_detection_zones()
        self.zone_detections = {}  # Store detections per zone
        self.confidence_threshold = 0.9  # Very high confidence for static zones
        self.detection_history = {}  # Track detection history per zone
        self.stability_frames = 5  # Frames needed for stable detection
        
    def _create_detection_zones(self):
        """Create static detection zones across the frame"""
        zones = []
        zone_width = self.frame_width // 3
        zone_height = self.frame_height // 3
        
        # Create 3x3 grid of detection zones
        for row in range(3):
            for col in range(3):
                x = col * zone_width
                y = row * zone_height
                zone = {
                    'id': f'zone_{row}_{col}',
                    'x': x,
                    'y': y,
                    'width': zone_width,
                    'height': zone_height,
                    'center_x': x + zone_width // 2,
                    'center_y': y + zone_height // 2
                }
                zones.append(zone)
        return zones
        
    def update_zones(self, detections):
        """Update static zones with new detections"""
        if not detections:
            return self.get_stable_detections()
            
        # Clear current zone detections
        current_zone_detections = {zone['id']: [] for zone in self.detection_zones}
        
        # Assign detections to zones based on center point
        for detection in detections:
            if detection['confidence'] < self.confidence_threshold:
                continue
                
            x, y, w, h = detection['box']
            center_x = x + w / 2
            center_y = y + h / 2
            
            # Find which zone this detection belongs to
            for zone in self.detection_zones:
                if (zone['x'] <= center_x <= zone['x'] + zone['width'] and
                    zone['y'] <= center_y <= zone['y'] + zone['height']):
                    current_zone_detections[zone['id']].append(detection)
                    break
        
        # Update detection history for stability
        for zone_id, zone_detections in current_zone_detections.items():
            if zone_id not in self.detection_history:
                self.detection_history[zone_id] = []
            
            # Add current frame detections
            self.detection_history[zone_id].append(zone_detections)
            
            # Keep only recent history
            if len(self.detection_history[zone_id]) > self.stability_frames:
                self.detection_history[zone_id].pop(0)
        
        return self.get_stable_detections()
    
    def get_stable_detections(self):
        """Get stable detections from zones that have consistent detection history"""
        stable_detections = []
        
        for zone_id, history in self.detection_history.items():
            if len(history) < self.stability_frames:
                continue
                
            # Check if zone has consistent detections
            detection_count = sum(1 for frame_detections in history if frame_detections)
            stability_ratio = detection_count / len(history)
            
            if stability_ratio >= 0.6:  # 60% of frames must have detections
                # Get the most recent detection from this zone
                for frame_detections in reversed(history):
                    if frame_detections:
                        # Use the highest confidence detection from this zone
                        best_detection = max(frame_detections, key=lambda d: d['confidence'])
                        stable_detections.append(best_detection)
                        break
        
        return stable_detections
    
    def draw_zones(self, frame):
        """Draw detection zones on frame for visualization"""
        for zone in self.detection_zones:
            # Draw zone boundaries
            cv2.rectangle(frame, 
                         (zone['x'], zone['y']), 
                         (zone['x'] + zone['width'], zone['y'] + zone['height']), 
                         (100, 100, 100), 1)
            
            # Draw zone ID
            cv2.putText(frame, zone['id'], 
                       (zone['x'] + 5, zone['y'] + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        return frame
        
        for distance, track_id, detection_idx in all_matches:
            if track_id not in used_tracks and detection_idx not in used_detections:
                matched_pairs.append((track_id, detection_idx))
                used_tracks.add(track_id)
                used_detections.add(detection_idx)
                unmatched_detections.remove(detection_idx)
                unmatched_tracks.remove(track_id)
        
        # Update matched tracks with smoothing
        for track_id, detection_idx in matched_pairs:
            center_x, center_y, detection = new_centers[detection_idx]
            track = self.tracked_objects[track_id]
            
            # Store velocity for prediction
            if 'velocity_x' not in track:
                track['velocity_x'] = 0
                track['velocity_y'] = 0
            else:
                # Calculate velocity based on position change
                track['velocity_x'] = center_x - track['smooth_x']
                track['velocity_y'] = center_y - track['smooth_y']
            
            # Smooth the position with velocity prediction
            track['smooth_x'] = self.smoothing_factor * track['smooth_x'] + (1 - self.smoothing_factor) * center_x
            track['smooth_y'] = self.smoothing_factor * track['smooth_y'] + (1 - self.smoothing_factor) * center_y
            
            # Smooth the box dimensions as well for stable reshaping
            x, y, w, h = detection['box']
            if 'smooth_w' not in track:
                track['smooth_w'] = w
                track['smooth_h'] = h
            else:
                track['smooth_w'] = self.smoothing_factor * track['smooth_w'] + (1 - self.smoothing_factor) * w
                track['smooth_h'] = self.smoothing_factor * track['smooth_h'] + (1 - self.smoothing_factor) * h
            
            # Update detection info with smoothed dimensions
            track['box'] = [
                track['smooth_x'] - track['smooth_w']/2,
                track['smooth_y'] - track['smooth_h']/2,
                track['smooth_w'], track['smooth_h']
            ]
            track['label'] = detection['label']
            track['confidence'] = detection['confidence']
            track['class_id'] = detection['class_id']
            track['last_seen'] = time.time()
        
        # Create new tracks for unmatched detections (strict duplicate prevention)
        for detection_idx in unmatched_detections:
            center_x, center_y, detection = new_centers[detection_idx]
            x, y, w, h = detection['box']
            
            # Check if this detection is too close to any existing track (prevent duplicates)
            too_close = False
            for existing_track in self.tracked_objects.values():
                existing_center = (existing_track['smooth_x'], existing_track['smooth_y'])
                distance = np.sqrt((existing_center[0] - center_x)**2 + (existing_center[1] - center_y)**2)
                if distance < 50 and existing_track.get('class_id', 0) == detection.get('class_id', 0):
                    too_close = True
                    break
            
            if not too_close:
                self.tracked_objects[self.next_id] = {
                    'smooth_x': center_x,
                    'smooth_y': center_y,
                    'smooth_w': w,
                    'smooth_h': h,
                    'box': [x, y, w, h],
                    'label': detection['label'],
                    'confidence': detection['confidence'],
                    'class_id': detection['class_id'],
                    'last_seen': time.time()
                }
                self.next_id += 1
        
        # Remove old tracks that haven't been seen recently
        current_time = time.time()
        tracks_to_remove = []
        for track_id, track in self.tracked_objects.items():
            if current_time - track['last_seen'] > 3.0:  # Remove after 3 seconds
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracked_objects[track_id]
        
        return list(self.tracked_objects.values())

class YOLODetector:
    def __init__(self, model_name="yolo11m.pt", use_fallback=False):
        self.model = None
        self.model_name = model_name
        self.confidence_threshold = 0.05  # Lower threshold to detect more people
        # Initialize object tracker for responsive detection
        self.tracker = ObjectTracker(
            smoothing_factor=0.7,  # Reduced smoothing for more responsive detection
            max_distance=100,
            frame_skip_interval=1,   # Process EVERY frame for maximum speed
            grace_period_ms=150     # Shorter grace period for faster response
        )
        self.use_fallback = use_fallback
        # ULTRA-HIGH FREQUENCY: No frame skipping - maximum detection speed
        self.detection_counter = 0
        
        # Enhanced detection stabilization for flicker-free boxes
        self.previous_detections = []
        self.persistent_detections = {}  # Track detections across frames
        self.smoothing_factor = 0.6  # Reduced smoothing for more responsive detection
        self.min_confidence_diff = 0.15  # Minimum confidence change to update detection
        self.detection_persistence = 10  # Shorter persistence for more responsive detection
        self.min_confidence_for_new = 0.3  # Lower threshold for new detections to catch more people
        
        # Motion detection for screen stability
        self.prev_frame = None
        self.motion_threshold = 30.0  # Threshold for detecting significant motion
        self.frame_buffer = []  # Buffer to store recent frames
        self.buffer_size = 3  # Number of frames to buffer
        self.motion_detected = False
        
        # GPU performance optimization settings
        self.gpu_batch_size = 1  # Batch size for GPU processing
        self.gpu_memory_fraction = 0.8  # Use 80% of GPU memory
        self.enable_tensorrt = True  # Enable TensorRT optimization if available
        
        # Check for CUDA-enabled GPU with enhanced detection
        import torch
        if torch.cuda.is_available() and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.device = "cuda"
            # Enable maximum GPU performance
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
            torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster computation
            torch.backends.cudnn.allow_tf32 = True
            # Set GPU to maximum performance mode
            if torch.cuda.device_count() > 0:
                torch.cuda.set_device(0)  # Use first GPU
                torch.cuda.empty_cache()  # Clear GPU cache
                print(f"[INFO] GPU optimization enabled - Device: {torch.cuda.get_device_name(0)}")
                print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            self.device = "cpu"
        print(f"[INFO] Using device: {self.device}")
        
        # YOLO COCO class definitions
        self.class_names = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
            6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
            11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
            16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
            22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
            27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
            32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
            36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
            40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
            46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
            51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
            56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
            61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
            67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
            72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
            77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
        }
        
        # Define animal classes for color coding
        self.animal_classes = {14, 15, 16, 17, 18, 19, 20, 21, 22, 23}  # bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
        
        # Initialize OpenCV face detection as working fallback
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            print("[INFO] OpenCV face detection initialized as fallback")
        except Exception as e:
            print(f"[WARNING] OpenCV face detection failed: {e}")
            self.face_cascade = None
        
        if not use_fallback:
            self.initialize_model()
        else:
            print("[INFO] YOLO detection disabled")
            self.model = None
    
    def initialize_model(self):
        """Initialize YOLO model exactly like reference code with Windows workarounds"""
        try:
            # Set comprehensive environment variables to bypass Windows security issues
            os.environ['YOLO_VERBOSE'] = 'False'
            os.environ['ULTRALYTICS_OFFLINE'] = '1'
            os.environ['ULTRALYTICS_DISABLE_TELEMETRY'] = '1'
            os.environ['ULTRALYTICS_DISABLE_UPDATE_CHECK'] = '1'
            os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
            
            # Additional Windows security fixes
            import tempfile
            os.environ['ULTRALYTICS_CONFIG_DIR'] = tempfile.gettempdir()
            os.environ['TORCH_HOME'] = tempfile.gettempdir()
            os.environ['HF_HOME'] = tempfile.gettempdir()
            
            # Import YOLO here to avoid early initialization issues
            from ultralytics import YOLO
            
            # Fix PyTorch security restrictions for Windows
            import torch
            torch.serialization._use_new_zipfile_serialization = False
            
            # Monkey patch torch.load to use weights_only=False
            original_load = torch.load
            def patched_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            torch.load = patched_load
            
            print(f"[INFO] Loading YOLO model: {self.model_name}")
            
            # Load the YOLO model with maximum performance optimization
            self.model = YOLO(self.model_name)
            
            # Move model to GPU with maximum performance settings
            if self.device == "cuda":
                self.model.to(self.device)
                
                # Set GPU memory management for maximum performance
                torch.cuda.set_per_process_memory_fraction(self.gpu_memory_fraction)
                torch.cuda.empty_cache()
                
                # Enable mixed precision for faster inference
                self.model.model.half()  # Convert to FP16 for speed
                # Optimize model for inference
                self.model.model.eval()
                
                # Enable compilation for maximum speed (PyTorch 2.0+)
                try:
                    self.model.model = torch.compile(self.model.model, mode="max-autotune")
                    print("[INFO] Model compiled with max-autotune for maximum speed")
                except:
                    print("[INFO] Model compilation not available, using standard optimization")
                
                # Warm up the model with multiple dummy inferences for optimal GPU state
                dummy_input = torch.randn(1, 3, 640, 640).to(self.device).half()
                with torch.no_grad():
                    for _ in range(3):  # Multiple warmup runs
                        _ = self.model.model(dummy_input)
                torch.cuda.synchronize()  # Ensure GPU is ready
                
                # Print GPU utilization info
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_allocated = torch.cuda.memory_allocated() / 1024**3
                print(f"[INFO] Model moved to GPU with FP16 optimization and warmed up")
                print(f"[INFO] GPU Memory allocated: {gpu_allocated:.1f}/{gpu_memory:.1f} GB")
            
            print("[INFO] YOLO model initialized with maximum performance settings")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize YOLO model: {e}")
            print("[INFO] YOLO detection will be disabled")
            self.model = None
            return False
    
    def detect_objects(self, frame):
        """Optimized detection with ChatGPT's anti-flicker recommendations"""
        if self.model is None:
            return []
            
        try:
            # Real-time inference - no FPS limiting for maximum responsiveness
            current_time = time.time()
            if not hasattr(self, 'last_inference_time'):
                self.last_inference_time = 0
            if not hasattr(self, 'last_detections'):
                self.last_detections = []
            
            # Process every frame for real-time detection (removed FPS throttling)
            
            # Run YOLO inference with maximum speed optimizations
            # Aggressive frame resizing for real-time performance
            height, width = frame.shape[:2]
            # Use smaller resolution for maximum speed (320px for real-time)
            target_size = 320  # Smaller size for faster processing
            if width > target_size or height > target_size:
                if width > height:
                    scale = target_size / width
                    new_width = target_size
                    new_height = int(height * scale)
                else:
                    scale = target_size / height
                    new_height = target_size
                    new_width = int(width * scale)
                frame_resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            else:
                frame_resized = frame
                scale = 1.0
            
            # Use model.track() with maximum GPU performance optimization
            try:
                if self.device == "cuda":
                    # Maximum GPU performance tracking with FP16 and smaller image size
                    results = self.model.track(
                        frame_resized,
                        persist=True,
                        tracker='bytetrack.yaml',
                        conf=self.confidence_threshold,
                        iou=0.3,  # Lower IoU to allow overlapping detections in crowds
                        verbose=False,
                        imgsz=320,  # Smaller size for real-time processing
                        half=True,  # Use FP16 for maximum speed
                        device=self.device,
                        max_det=100  # Increase max detections for crowds
                    )
                else:
                    results = self.model.track(
                        frame_resized,
                        persist=True,
                        tracker='bytetrack.yaml',
                        conf=self.confidence_threshold,
                        iou=0.3,  # Lower IoU for crowd detection
                        verbose=False,
                        imgsz=320,  # Smaller size for real-time processing
                        max_det=100  # Increase max detections for crowds
                    )
            except:
                # Fallback to regular detection with GPU optimization
                if self.device == "cuda":
                    results = self.model(frame_resized, conf=self.confidence_threshold, verbose=False, show=False, imgsz=320, half=True, device=self.device, max_det=100)
                else:
                    results = self.model(frame_resized, conf=self.confidence_threshold, verbose=False, show=False, imgsz=320, max_det=100)
            
            detections = []
            # Get first result (like reference code: results[0])
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
                        
                        # Get class name from YOLO's built-in names (like reference)
                        class_name = self.model.names[class_id] if class_id in self.model.names else f'class_{class_id}'
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'label': class_name,
                            'track_id': track_id
                        }
                        detections.append(detection)
            
            self.last_detections = detections
            self.last_inference_time = current_time
            return detections
            
        except Exception as e:
            print(f"[ERROR] Detection failed: {e}")
            return getattr(self, 'last_detections', [])
    
    def detect(self, frame):
        """Simple YOLO detection matching reference code behavior"""
        if self.model is None:
            return []
        
        try:
            # Detect motion to adjust detection sensitivity
            self.detect_motion(frame)
            
            # Run YOLO model inference with maximum GPU performance and real-time optimization
            if self.device == "cuda":
                # Use optimized GPU inference with FP16 and smaller image size
                results = self.model(frame, verbose=False, show=False, half=True, device=self.device, imgsz=320, max_det=100)
            else:
                results = self.model(frame, verbose=False, show=False, imgsz=320, max_det=100)
            
            detections = []
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    # Extract box coordinates and info
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Use YOLO's built-in class names
                    class_name = self.model.names[class_id] if class_id < len(self.model.names) else f"class_{class_id}"
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence),
                        'class': class_name
                    })
            
            return detections
            
        except Exception as e:
            print(f"[ERROR] YOLO detection failed: {e}")
            return []
    
    def get_class_name(self, class_id):
        """Get human-readable class name from class ID."""
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"Class {class_id}"
    
    def detect_motion(self, frame):
        """
        Detect significant motion in the frame to adjust detection sensitivity
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame is not None:
            # Check if frame sizes match to prevent OpenCV size mismatch errors
            if self.prev_frame.shape == gray.shape:
                # Calculate frame difference
                diff = cv2.absdiff(self.prev_frame, gray)
                mean_diff = np.mean(diff)
                
                # Detect motion based on threshold
                self.motion_detected = mean_diff > self.motion_threshold
            else:
                # Frame size changed, reset motion detection
                self.motion_detected = False
                print(f"[INFO] Frame size changed from {self.prev_frame.shape} to {gray.shape}, resetting motion detection")
        else:
            self.motion_detected = False
            
        self.prev_frame = gray.copy()
        return self.motion_detected
    
    def get_stabilized_boxes(self, current_detections):
        """
        Enhanced coordinate smoothing with motion detection for stable detection.
        Reduces sensitivity during phone screen movement.
        """
        current_boxes = []
        
        # Extract current detection info
        for det in current_detections:
            curr_coords = det.xyxy[0].tolist()
            curr_conf = det.conf.item()
            curr_cls = int(det.cls.item())
            
            # Use higher threshold for more stable detections
            if curr_conf >= max(self.confidence_threshold, 0.6):
                current_boxes.append({
                    'coords': curr_coords,
                    'conf': curr_conf,
                    'cls': curr_cls
                })
        
        # Simple smoothing with previous detections
        if hasattr(self, 'prev_detections') and self.prev_detections:
            smoothed_boxes = []
            
            for curr_box in current_boxes:
                # Find closest previous detection of same class
                best_match = None
                min_distance = float('inf')
                
                curr_center = [(curr_box['coords'][0] + curr_box['coords'][2]) / 2,
                              (curr_box['coords'][1] + curr_box['coords'][3]) / 2] 
                
                for prev_box in self.prev_detections:
                    if prev_box['cls'] == curr_box['cls']:
                        prev_center = [(prev_box['coords'][0] + prev_box['coords'][2]) / 2,
                                      (prev_box['coords'][1] + prev_box['coords'][3]) / 2]
                        
                        distance = ((curr_center[0] - prev_center[0]) ** 2 + 
                                  (curr_center[1] - prev_center[1]) ** 2) ** 0.5
                        
                        if distance < min_distance and distance < 100:
                            min_distance = distance
                            best_match = prev_box
                
                if best_match:
                    # Apply adaptive smoothing based on motion detection
                    if self.motion_detected:
                        # During motion, use very strong smoothing to prevent flickering
                        smoothing = 0.98  # Ultra-high smoothing during motion
                    else:
                        # Normal smoothing when screen is stable
                        smoothing = 0.85
                    
                    smoothed_coords = [
                        smoothing * best_match['coords'][0] + (1 - smoothing) * curr_box['coords'][0],
                        smoothing * best_match['coords'][1] + (1 - smoothing) * curr_box['coords'][1],
                        smoothing * best_match['coords'][2] + (1 - smoothing) * curr_box['coords'][2],
                        smoothing * best_match['coords'][3] + (1 - smoothing) * curr_box['coords'][3]
                    ]
                    
                    smoothed_boxes.append({
                        'coords': smoothed_coords,
                        'conf': curr_box['conf'],
                        'cls': curr_box['cls'],
                        'class_name': self.get_class_name(curr_box['cls'])
                    })
                else:
                    # No match found, use current detection as-is
                    smoothed_boxes.append({
                        'coords': curr_box['coords'],
                        'conf': curr_box['conf'],
                        'cls': curr_box['cls'],
                        'class_name': self.get_class_name(curr_box['cls'])
                    })
            
            # Store current detections for next frame
            self.prev_detections = current_boxes.copy()
            return smoothed_boxes
        else:
            # First frame or no previous detections
            self.prev_detections = current_boxes.copy()
            return [{
                'coords': box['coords'],
                'conf': box['conf'],
                'cls': box['cls'],
                'class_name': self.get_class_name(box['cls'])
            } for box in current_boxes]
    
    def detect_and_draw_simple(self, frame):
        """Smart YOLO detection with motion detection and anti-flickering for static image stability"""
        if self.model is None:
            return frame
            
        try:
            # Initialize frame buffering, motion detection and VSync detection persistence
            if not hasattr(self, 'frame_buffer'):
                self.frame_buffer = []
                self.detection_cache = None
                self.last_detection_time = 0
                self.frame_skip_counter = 0
                self.detection_interval = 0.000000001  # Ultra-ultra-high frequency detection
                # VSync: Dynamic FPS detection and matching
                self.vsync_enabled = True
                self.frame_times = []
                self.detected_fps = 60  # Default FPS
                self.cache_duration = 1.0 / self.detected_fps  # Dynamic cache based on detected FPS
                self.prev_frame_gray = None
                self.motion_threshold = 50  # Lower threshold for more sensitive motion detection
                self.static_frame_count = 0
                self.max_static_frames = self.detected_fps * 3  # 3 seconds worth of frames
            
            current_time = time.time()
            
            # VSync: Dynamic FPS detection and synchronization
            if self.vsync_enabled:
                self.frame_times.append(current_time)
                # Keep only last 30 frame times for rolling average
                if len(self.frame_times) > 30:
                    self.frame_times.pop(0)
                
                # Calculate FPS every 10 frames
                if len(self.frame_times) >= 10:
                    time_diff = self.frame_times[-1] - self.frame_times[0]
                    if time_diff > 0:
                        calculated_fps = (len(self.frame_times) - 1) / time_diff
                        # Smooth FPS changes to avoid jitter
                        self.detected_fps = self.detected_fps * 0.9 + calculated_fps * 0.1
                        # Update cache duration and static frame limits based on detected FPS
                        self.cache_duration = 1.0 / max(self.detected_fps, 1)
                        self.max_static_frames = int(self.detected_fps * 3)  # 3 seconds worth
            
            # Convert frame to grayscale for motion detection
            import cv2
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate motion between frames
            motion_detected = True
            if self.prev_frame_gray is not None:
                frame_diff = cv2.absdiff(self.prev_frame_gray, gray_frame)
                motion_amount = cv2.sumElems(frame_diff)[0]
                
                if motion_amount < self.motion_threshold:
                    self.static_frame_count += 1
                    motion_detected = False
                else:
                    self.static_frame_count = 0
                    motion_detected = True
            
            self.prev_frame_gray = gray_frame.copy()
            
            # Smart frame processing: skip frames to reduce flickering and stop on static images
            self.frame_skip_counter += 1
            should_detect = ((self.frame_skip_counter >= self.detection_interval) or (self.detection_cache is None)) and \
                           (motion_detected or self.static_frame_count < self.max_static_frames)
            
            if should_detect:
                self.frame_skip_counter = 0
                
                # Run YOLO model inference with optimized settings
                if self.device == "cuda":
                    results = self.model(frame, verbose=False, show=False, half=True, device=self.device, imgsz=320, max_det=100)
                else:
                    results = self.model(frame, verbose=False, show=False, imgsz=320, max_det=100)
                
                if results and len(results) > 0 and results[0].boxes is not None:
                    # Cache the detection results with timestamp
                    self.detection_cache = {
                        'results': results[0],
                        'timestamp': current_time,
                        'annotated_frame': results[0].plot()
                    }
                    self.last_detection_time = current_time
                    return self.detection_cache['annotated_frame']
            
            # Use cached detections if available and recent
            if (self.detection_cache is not None and 
                (current_time - self.last_detection_time) < self.cache_duration):
                return self.detection_cache['annotated_frame']
            
            # Return original frame if no valid detections
            return frame
                
        except Exception as e:
            print(f"[ERROR] Detection failed: {e}")
            return frame
    
    def detect_and_draw_simple_with_data(self, frame):
        """Smart YOLO detection with motion detection and returns detection data for overlay mode"""
        if self.model is None:
            return frame, []
            
        try:
            # Initialize frame buffering, motion detection and TTL persistence (ChatGPT anti-flicker fix)
            if not hasattr(self, 'data_frame_buffer'):
                self.data_frame_buffer = []
                self.data_detection_cache = None
                self.data_last_detection_time = 0
                self.data_frame_skip_counter = 0
                self.data_detection_interval = 1/15  # 15 FPS inference as recommended by ChatGPT
                # TTL persistence: Hold detections for 250-300ms to prevent flicker
                self.TTL = 0.30  # 300ms TTL as recommended by ChatGPT
                self.last_boxes = []
                self.last_detection_timestamp = 0
                # VSync: Dynamic FPS detection and matching for data function
                self.data_vsync_enabled = True
                self.data_frame_times = []
                self.data_detected_fps = 60  # Default FPS
                self.data_prev_frame_gray = None
                self.data_motion_threshold = 50  # Lower threshold for more sensitive motion detection
                self.data_static_frame_count = 0
                self.data_max_static_frames = 90  # Allow more frames before stopping (3 seconds at 30fps)
            
            current_time = time.time()
            
            # VSync: Dynamic FPS detection and synchronization for data function
            if self.data_vsync_enabled:
                self.data_frame_times.append(current_time)
                # Keep only last 30 frame times for rolling average
                if len(self.data_frame_times) > 30:
                    self.data_frame_times.pop(0)
                
                # Calculate FPS every 10 frames
                if len(self.data_frame_times) >= 10:
                    time_diff = self.data_frame_times[-1] - self.data_frame_times[0]
                    if time_diff > 0:
                        calculated_fps = (len(self.data_frame_times) - 1) / time_diff
                        # Smooth FPS changes to avoid jitter
                        self.data_detected_fps = self.data_detected_fps * 0.9 + calculated_fps * 0.1
                        # Update cache duration and static frame limits based on detected FPS
                        self.data_cache_duration = 1.0 / max(self.data_detected_fps, 1)
                        self.data_max_static_frames = int(self.data_detected_fps * 3)  # 3 seconds worth
            
            # Convert frame to grayscale for motion detection
            import cv2
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate motion between frames
            motion_detected = True
            if self.data_prev_frame_gray is not None:
                # Check if frame sizes match to prevent OpenCV size mismatch errors
                if self.data_prev_frame_gray.shape == gray_frame.shape:
                    frame_diff = cv2.absdiff(self.data_prev_frame_gray, gray_frame)
                    motion_amount = cv2.sumElems(frame_diff)[0]
                    
                    if motion_amount < self.data_motion_threshold:
                        self.data_static_frame_count += 1
                        motion_detected = False
                    else:
                        self.data_static_frame_count = 0
                        motion_detected = True
                else:
                    # Frame size changed, reset motion detection
                    motion_detected = True
                    self.data_static_frame_count = 0
                    print(f"[INFO] Data frame size changed from {self.data_prev_frame_gray.shape} to {gray_frame.shape}, resetting motion detection")
            
            self.data_prev_frame_gray = gray_frame.copy()
            
            # ChatGPT TTL inference: Run YOLO at 15 FPS, hold detections for 300ms
            should_detect = (current_time - self.data_last_detection_time >= self.data_detection_interval) and \
                           (motion_detected or self.data_static_frame_count < self.data_max_static_frames)
            
            if should_detect:
                # Run YOLO model inference with optimized settings
                if self.device == "cuda":
                    results = self.model(frame, verbose=False, show=False, half=True, device=self.device, imgsz=320, max_det=100)
                else:
                    results = self.model(frame, verbose=False, show=False, imgsz=320, max_det=100)
                
                if results and len(results) > 0 and results[0].boxes is not None:
                    detections = results[0].boxes
                    
                    # Use YOLO's built-in colorful visualization
                    annotated_frame = results[0].plot()
                    
                    # Get detection data for Electron overlay and store in TTL cache
                    detection_data = []
                    for box in detections:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        # Only include person detections (class 0) with sufficient confidence
                        if cls == 0 and conf >= self.confidence_threshold:
                            detection_data.append({
                                'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2),
                                'confidence': float(conf), 'class': 'person'
                            })
                    
                    # Output detection data as JSON for Electron to parse
                    if detection_data:
                        import json
                        print(f"DETECTION_DATA:{json.dumps(detection_data)}", flush=True)
                    
                    # Update TTL cache with new detections
                    self.last_boxes = detection_data
                    self.last_detection_timestamp = current_time
                    self.data_last_detection_time = current_time
                    return frame, detection_data  # Return original frame, not annotated
            
            # ChatGPT TTL persistence: Use cached detections if within TTL window
            if (self.last_boxes and 
                (current_time - self.last_detection_timestamp) < self.TTL):
                # Return frame with persistent detections (no flicker)
                return frame, self.last_boxes
            
            # TTL expired or no detections - return empty
            return frame, []
                
        except Exception as e:
            print(f"[ERROR] Detection failed: {e}")
            return frame, []
    
    def detect_and_draw_static_zones(self, frame):
        """Detection with static zones - ultra-stable detection areas"""
        if self.model is None:
            return frame
            
        try:
            # Get raw detections from YOLO
            raw_detections = self.detect_objects(frame)
            
            # Convert to static zone format
            zone_detections = []
            for detection in raw_detections:
                x1, y1, x2, y2 = detection['bbox']
                zone_detections.append({
                    'box': [x1, y1, x2-x1, y2-y1],  # Convert to x,y,w,h format
                    'label': detection['label'],
                    'confidence': detection['confidence'],
                    'class_id': detection['class_id']
                })
            
            # Update static zones with new detections
            stable_detections = self.static_zones.update_zones(zone_detections)
            
            # Draw zone boundaries for visualization
            frame_with_zones = self.static_zones.draw_zones(frame.copy())
            
            # Draw stable detections
            frame_with_detections = self.draw_detections(frame_with_zones, stable_detections)
            return frame_with_detections
                
        except Exception as e:
            print(f"[ERROR] Static zone detection failed: {e}")
            return frame
    
    def fallback_detection(self, frame):
        """Fallback detection using OpenCV face detection"""
        if self.face_cascade is None:
            return []
            
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            detections = []
            for (x, y, w, h) in faces:
                detection = {
                    'bbox': [int(x), int(y), int(x + w), int(y + h)],
                    'confidence': 0.8,  # Fixed confidence for face detection
                    'class_id': 0,  # person class
                    'label': 'person'
                }
                detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"[ERROR] Face detection failed: {e}")
            return []
    
    def get_detection_color(self, class_id):
        """Get color based on object class - green for persons, blue for animals, red for others"""
        if class_id == 0:  # person
            return (0, 255, 0)  # Green
        elif class_id in self.animal_classes:  # animals
            return (255, 0, 0)  # Blue (BGR format)
        else:  # other objects
            return (0, 0, 255)  # Red
    
    def draw_detections(self, frame, detections):
        """Draw detection boxes and labels on frame"""
        for detection in detections:
            # Handle both bbox and box formats
            if 'bbox' in detection:
                x1, y1, x2, y2 = detection['bbox']
                x, y, w, h = x1, y1, x2-x1, y2-y1
            else:
                x, y, w, h = detection['box']
            
            label = detection['label']
            confidence = detection['confidence']
            class_id = detection['class_id']
            
            # Convert to integers for OpenCV
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            # Get color based on class type
            color = self.get_detection_color(class_id)
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            
            # Draw label with background
            label_text = f"{label}: {confidence:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - text_height - 10), (x + text_width, y), color, -1)
            cv2.putText(frame, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame

class ScreenCapture:
    def __init__(self, region=None):
        self.region = region  # (x, y, width, height)
        
    def capture_screen(self):
        """Capture screen or specific region"""
        try:
            import mss
            
            with mss.mss() as sct:
                if self.region:
                    x, y, width, height = self.region
                    monitor = {"top": y, "left": x, "width": width, "height": height}
                else:
                    monitor = sct.monitors[1]  # Primary monitor
                
                screenshot = sct.grab(monitor)
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                return frame
        except ImportError:
            print("[ERROR] mss library not found. Install with: pip install mss")
            return None
        except Exception as e:
            print(f"[ERROR] Screen capture failed: {e}")
            return None

class WindowOverlay:
    def __init__(self, window_title="Argus Phone Mirror"):
        self.window_title = window_title
        self.hwnd = None
        self.overlay_hwnd = None
        
    def find_window(self):
        """Find the target window"""
        try:
            def enum_windows_callback(hwnd, windows):
                if win32gui.IsWindowVisible(hwnd):
                    window_text = win32gui.GetWindowText(hwnd)
                    if self.window_title.lower() in window_text.lower():
                        windows.append(hwnd)
                return True
            
            windows = []
            win32gui.EnumWindows(enum_windows_callback, windows)
            
            if windows:
                self.hwnd = windows[0]
                return True
            else:
                print(f"[WARNING] Window '{self.window_title}' not found")
                return False
                
        except Exception as e:
            print(f"[ERROR] Failed to find window: {e}")
            return False
    
    def create_overlay(self):
        """Create a transparent overlay window"""
        if not self.hwnd:
            return False
            
        try:
            # Get target window position and size
            rect = win32gui.GetWindowRect(self.hwnd)
            x, y, right, bottom = rect
            width = right - x
            height = bottom - y
            
            # Create overlay window class
            wc = win32gui.WNDCLASS()
            wc.lpfnWndProc = win32gui.DefWindowProc
            wc.lpszClassName = "YOLOOverlay"
            wc.hCursor = win32gui.LoadCursor(0, win32con.IDC_ARROW)
            
            try:
                win32gui.RegisterClass(wc)
            except:
                pass  # Class might already be registered
            
            # Create the overlay window
            self.overlay_hwnd = win32gui.CreateWindowEx(
                win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_TOPMOST,
                "YOLOOverlay",
                "YOLO Overlay",
                win32con.WS_POPUP,
                x, y, width, height,
                0, 0, 0, None
            )
            
            if self.overlay_hwnd:
                # Make window transparent
                win32gui.SetLayeredWindowAttributes(self.overlay_hwnd, 0, 200, win32con.LWA_ALPHA)
                win32gui.ShowWindow(self.overlay_hwnd, win32con.SW_SHOW)
                return True
                
        except Exception as e:
            print(f"[ERROR] Failed to create overlay: {e}")
            
        return False
    
    def get_window_color(self, class_id):
        """Get Windows API RGB color based on object class"""
        if class_id == 0:  # person
            return win32api.RGB(0, 255, 0)  # Green
        elif class_id in {14, 15, 16, 17, 18, 19, 20, 21, 22, 23}:  # animals
            return win32api.RGB(0, 0, 255)  # Blue
        else:  # other objects
            return win32api.RGB(255, 0, 0)  # Red
    
    def draw_detections_on_window(self, detections):
        """Draw detection boxes directly on the target window"""
        if not self.hwnd:
            return
            
        try:
            # Get window device context
            hdc = win32gui.GetWindowDC(self.hwnd)
            
            # Create transparent brush (hollow brush)
            brush = win32gui.GetStockObject(win32con.HOLLOW_BRUSH)
            old_brush = win32gui.SelectObject(hdc, brush)
            
            # Set transparent background
            win32gui.SetBkMode(hdc, win32con.TRANSPARENT)
            
            for detection in detections:
                x, y, w, h = detection['box']
                label = detection['label']
                confidence = detection['confidence']
                class_id = detection.get('class_id', 0)
                
                # Get color based on class
                color = self.get_window_color(class_id)
                
                # Create pen with appropriate color
                pen = win32gui.CreatePen(win32con.PS_SOLID, 3, color)
                old_pen = win32gui.SelectObject(hdc, pen)
                
                # Set text color to match box color
                win32gui.SetTextColor(hdc, color)
                
                # Draw rectangle with transparent fill
                win32gui.Rectangle(hdc, int(x), int(y), int(x + w), int(y + h))
                
                # Draw label using DrawText
                text = f"{label}: {confidence:.2f}"
                rect = (int(x), int(y - 25), int(x + 200), int(y))
                win32gui.DrawText(hdc, text, -1, rect, 0)
                
                # Cleanup pen
                win32gui.SelectObject(hdc, old_pen)
                win32gui.DeleteObject(pen)
            
            # Cleanup brush
            win32gui.SelectObject(hdc, old_brush)
            win32gui.ReleaseDC(self.hwnd, hdc)
            
        except Exception as e:
            print(f"[ERROR] Failed to draw on window: {e}")
    
    def cleanup(self):
        """Clean up overlay window"""
        if self.overlay_hwnd:
            try:
                win32gui.DestroyWindow(self.overlay_hwnd)
            except:
                pass
            self.overlay_hwnd = None

class WindowCapture:
    def __init__(self, window_title="Argus Phone Mirror"):
        self.window_title = window_title
        self.hwnd = None
        
    def find_window(self):
        """Find window by title"""
        try:
            import win32gui
            import win32con
            
            def enum_windows_callback(hwnd, windows):
                if win32gui.IsWindowVisible(hwnd) and self.window_title in win32gui.GetWindowText(hwnd):
                    windows.append(hwnd)
                return True
            
            windows = []
            win32gui.EnumWindows(enum_windows_callback, windows)
            
            if windows:
                self.hwnd = windows[0]
                print(f"[INFO] Found window: {win32gui.GetWindowText(self.hwnd)}")
                return True
            else:
                print(f"[WARNING] Window '{self.window_title}' not found")
                return False
                
        except ImportError:
            print("[ERROR] pywin32 library not found. Install with: pip install pywin32")
            return False
        except Exception as e:
            print(f"[ERROR] Window search failed: {e}")
            return False
    
    def capture_window(self):
        """Capture window content"""
        if not self.hwnd:
            if not self.find_window():
                return None
                
        try:
            import win32gui
            import win32ui
            import win32con
            
            # Get window dimensions
            left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
            width = right - left
            height = bottom - top
            # print(f"[DEBUG] Window capture dimensions: {width}x{height}", flush=True)
            
            # Get window device context
            hwndDC = win32gui.GetWindowDC(self.hwnd)
            mfcDC = win32ui.CreateDCFromHandle(hwndDC)
            saveDC = mfcDC.CreateCompatibleDC()
            
            # Create bitmap
            saveBitMap = win32ui.CreateBitmap()
            saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
            saveDC.SelectObject(saveBitMap)
            
            # Copy window content
            saveDC.BitBlt((0, 0), (width, height), mfcDC, (0, 0), win32con.SRCCOPY)
            
            # Convert to numpy array
            bmpinfo = saveBitMap.GetInfo()
            bmpstr = saveBitMap.GetBitmapBits(True)
            
            frame = np.frombuffer(bmpstr, dtype='uint8')
            frame.shape = (height, width, 4)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
     
            win32gui.DeleteObject(saveBitMap.GetHandle())
            saveDC.DeleteDC()
            mfcDC.DeleteDC()
            win32gui.ReleaseDC(self.hwnd, hwndDC)
            
            return frame
            
        except ImportError:
            print("[ERROR] pywin32 library not found. Install with: pip install pywin32")
            return None
        except Exception as e:
            print(f"[ERROR] Window capture failed: {e}")
            return None

def main():
    parser = argparse.ArgumentParser(description='Argus YOLO Detection')
    parser.add_argument('--source', default='screen', help='Source: screen, camera, window, or file path')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--region', default=None, help='Screen region: x,y,width,height')
    parser.add_argument('--window-title', default='Argus Phone Mirror', help='Window title to capture')
    parser.add_argument('--output', default=None, help='Output video file path')
    parser.add_argument('--show', action='store_true', help='Display detection window')
    parser.add_argument('--yolo-scrcpy-mode', action='store_true', help='YOLO scrcpy window mode')
    parser.add_argument('--overlay', action='store_true', help='Draw detections directly on target window')
    # Fallback detection removed - using pure YOLO only
    
    args = parser.parse_args()
    
    # Parse region if provided, otherwise use default target region
    region = None
    if args.region:
        try:
            region = list(map(int, args.region.split(',')))
            print(f"[INFO] Detection region: {region}")
        except:
            print(f"[ERROR] Invalid region format: {args.region}")
            sys.exit(1)
    else:
        # Default to the final YOLO coordinates: (139, 119, 1498, 936)
        region = [139, 119, 1498, 936]
        print(f"[INFO] Using final YOLO target region: {region}")
    
    # Initialize detector
    detector = YOLODetector(use_fallback=False)
    
    # Initialize capture
    if getattr(args, 'yolo_scrcpy_mode', False):
        # YOLO scrcpy mode - use camera
        cap = cv2.VideoCapture(args.camera)
        capture = None
        window_capture = None
        window_overlay = None
        print(f"[INFO] YOLO scrcpy mode - using camera {args.camera}")
    elif args.source == 'screen':
        capture = ScreenCapture(region)
        cap = None
        window_capture = None
        window_overlay = None
    elif args.source == 'window':
        window_capture = WindowCapture(args.window_title)
        capture = None
        cap = None
        print(f"[INFO] Window capture mode - targeting '{args.window_title}'")
        
        # Initialize overlay if requested
        window_overlay = None
        if args.overlay:
            window_overlay = WindowOverlay(args.window_title)
            if window_overlay.find_window():
                print(f"[INFO] Overlay mode enabled for '{args.window_title}'")
            else:
                print(f"[WARNING] Could not find window for overlay mode")
                window_overlay = None
    else:
        cap = cv2.VideoCapture(args.source if args.source != 'camera' else args.camera)
        capture = None
        window_capture = None
        window_overlay = None
    
    # Initialize video writer if output specified
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(args.output, fourcc, 20.0, (640, 480))
    
    print("[INFO] Starting SAR detection...")
    print("[INFO] Press 'q' to quit")
    
    try:
        while True:
            # Capture frame
            if capture:  # Screen capture
                frame = capture.capture_screen()
                if frame is None:
                    break
            elif window_capture:  # Window capture
                frame = window_capture.capture_window()
                if frame is None:
                    break
            else:  # Camera/video capture
                ret, frame = cap.read()
                if not ret:
                    break
            
            # ULTRA-HIGH FREQUENCY: No frame skipping - detect EVERY millisecond
            # Get stabilized detections and annotated frame - IMMEDIATE PROCESSING
            frame_with_detections, stabilized_detections = detector.detect_and_draw_simple_with_data(frame.copy())
            
            # For overlay mode, use the same stabilized detections for window overlay
            if window_overlay and stabilized_detections:
                # Convert stabilized format for overlay drawing
                overlay_detections = []
                for d in stabilized_detections:
                    x1, y1, x2, y2 = map(int, d['coords'])
                    overlay_detections.append({
                    'box': [x1, y1, x2-x1, y2-y1],  # Convert to x,y,w,h format
                    'label': detector.get_class_name(d['cls']),
                    'confidence': d['conf'],
                    'class_id': d['cls']
                })
                if overlay_detections:
                    detection_summary = ", ".join([f"{d['label']}({d['confidence']:.2f})" for d in overlay_detections])
                    # Removed detection logging to clean up interface
                    window_overlay.draw_detections_on_window(overlay_detections)
            
            # Save frame if output specified
            if out:
                out.write(frame_with_detections)
            
            # Display if requested or in yolo-scrcpy-mode (but not if overlay mode)
            if (args.show or getattr(args, 'yolo_scrcpy_mode', False)) and not args.overlay:
                cv2.imshow('Argus SAR Detection', frame_with_detections)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            elif args.overlay:
                # In overlay mode, just check for 'q' key without showing window
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # No delay for 60 FPS processing - let hardware handle the framerate
    
    except KeyboardInterrupt:
        print("\n[INFO] Detection stopped by user")
    
    finally:
        # Cleanup
        if cap:
            cap.release()
        if out:
            out.release()
        if window_overlay:
            window_overlay.cleanup()
        cv2.destroyAllWindows()
        print("[INFO] SAR detection terminated")

if __name__ == "__main__":
    main()