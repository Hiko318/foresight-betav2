#!/usr/bin/env python3
"""
Foresight - YOLO Detection Script
This script handles SAR (Search and Rescue) mode detection using YOLO
"""

# Force UTF-8 console to avoid UnicodeEncodeError
import os, sys
os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import cv2
import numpy as np
import argparse
import time
import sys
import os
from pathlib import Path
from collections import defaultdict, deque
import uuid

# Ensure DeepFace weights directory exists and startup self-test (ASCII-only)
try:
    from deepface import DeepFace
    os.makedirs(r"C:\Users\Asus\.deepface\weights", exist_ok=True)
    print("[FORESIGHT][DeepFace] Testing ArcFace...")
    try:
        DeepFace.build_model("ArcFace")
        print("[FORESIGHT][DeepFace] OK: ArcFace model loaded.")
    except Exception as e:
        print("[FORESIGHT][DeepFace] FAIL:", repr(e))
        if "arcface_weights.h5" in str(e).lower():
            print("[FORESIGHT][DeepFace] Missing weights. Place file at C:\\Users\\Asus\\.deepface\\weights\\arcface_weights.h5")
        print("[FORESIGHT][DeepFace] Continuing without DeepFace.")
except Exception as e:
    # DeepFace import itself failed; continue without DeepFace
    print("[FORESIGHT][DeepFace] FAIL:", repr(e))
    print("[FORESIGHT][DeepFace] Continuing without DeepFace.")

# =============================================================
# Foresight Face Save & DB Match System (ArcFace, cosine)
#
# Save directories and toggles:
# - SAVE_DIR: where new detected faces are saved
# Minimal saver constants (no ML, no dedupe)
# - DB_DIR:   canonical DB of known faces (used by DeepFace.find)
# - BYPASS_DEEPFACE: if True, skip DB check and save immediately
# - FORCE_ALWAYS_SAVE: if True, always save even if DB match
# - FRAMES_REQUIRED: consecutive frames per track before running face job
# - MIN_FACE_SIZE: minimum crop size to attempt face extraction
# - LOG_PREFIX: prefix added to all logs for easier filtering
#
# Thresholds:
# - DEEPFACE_MODEL: "ArcFace"
# - DEEPFACE_METRIC: "cosine"
# - DEEPFACE_THRESH: 0.33 (<= is considered a match)
#
# Notes:
# - Faces < MIN_FACE_SIZE are skipped
# - Retain only 1 representative face (middle of burst)
# - Do not upgrade TensorFlow/Keras; ignore their deprecation warnings
# =============================================================
SAVE_DIR = r"C:\Users\Asus\Desktop\Detected"
PER_TRACK_COOLDOWN_S = 6
EMB_DB   = r"C:\Users\Asus\Desktop\Detected\faces.db"
FACE_DETECTOR_BACKEND = "retinaface"
DEEPFACE_MODEL, DEEPFACE_METRIC, DEEPFACE_THRESH = "ArcFace", "cosine", 0.33
FRAMES_REQUIRED   = 5
CAPTURE_TIMEOUT_S = 5.0
SAME_PERSON_THRESH = 0.40  # tune 0.35–0.50
DB_MATCH_THRESH    = 0.42
MIN_FACE_SIZE = 40
MIN_FACE_WH = 40
BYPASS_DEEPFACE, FORCE_ALWAYS_SAVE = True, False
LOG_PREFIX = "[FORESIGHT]"
GLOBAL_WINDOW_OVERLAY = None

try:
    os.makedirs(SAVE_DIR, exist_ok=True)
    # Global per-track last save timestamps
    track_last_save = globals().get('track_last_save', {})
except Exception as _e:
    print(f"{LOG_PREFIX}[SETUP]dir_err:{_e}", flush=True)
try:
    import win32gui
    import win32ui
    import win32con
    import win32api
except ImportError:
    print("[WARNING] pywin32 not available - window overlay features disabled")

# SQLite (faces DB)
try:
    import sqlite3
    from datetime import datetime
    _conn = sqlite3.connect(EMB_DB)
    _cur  = _conn.cursor()
    _cur.execute("""CREATE TABLE IF NOT EXISTS faces(
      id INTEGER PRIMARY KEY, ts REAL, path TEXT, emb BLOB
    )""")
    _conn.commit()
except Exception as e:
    print(f"{LOG_PREFIX} [DB][ERR] {repr(e)}")
    _conn = None
    _cur = None

# InsightFace (GPU preferred)
app = None
try:
    from insightface.app import FaceAnalysis
    providers = ["CUDAExecutionProvider","CPUExecutionProvider"]
    try:
        app = FaceAnalysis(name="buffalo_l", providers=providers)
    except Exception:
        app = FaceAnalysis(name="buffalo_l")
    try:
        app.prepare(ctx_id=0, det_size=(640,640))
    except Exception:
        try:
            app.prepare(ctx_id=-1, det_size=(640,640))
        except Exception as e:
            print(f"{LOG_PREFIX} [INSIGHTFACE][ERR] {repr(e)}")
            app = None
except Exception as e:
    print(f"{LOG_PREFIX} [INSIGHTFACE][IMP][ERR] {repr(e)}")
    app = None

def _cosine_dist(a,b):
    try:
        d = (np.linalg.norm(a)+1e-8)*(np.linalg.norm(b)+1e-8)
        return 1.0 - float(np.dot(a,b)/d)
    except Exception:
        return 1.0

def _embed_one(bgr):
    try:
        if app is None:
            return None
        fs = app.get(bgr)
        if not fs:
            return None
        f = max(fs, key=lambda x: getattr(x, 'det_score', 0.0))
        return f.embedding.astype(np.float32)
    except Exception as e:
        print(f"{LOG_PREFIX} [EMB][ERR] {repr(e)}")
        return None

def _db_any_match(emb, thr):
    try:
        if _cur is None:
            return False
        for (blob,) in _cur.execute("SELECT emb FROM faces ORDER BY id DESC LIMIT 500"):
            try:
                if _cosine_dist(emb, np.frombuffer(blob, dtype=np.float32)) <= thr:
                    return True
            except Exception:
                continue
        return False
    except Exception as e:
        print(f"{LOG_PREFIX} [DB][CHECK][ERR] {repr(e)}")
        return False

def _db_any_match_in_folder(emb, thr):
    """Check for a match only among entries whose path is in SAVE_DIR."""
    try:
        if _cur is None:
            return False
        like_prefix = SAVE_DIR + '%'
        for (blob,) in _cur.execute("SELECT emb FROM faces WHERE path LIKE ? ORDER BY id DESC LIMIT 200", (like_prefix,)):
            try:
                if _cosine_dist(emb, np.frombuffer(blob, dtype=np.float32)) <= thr:
                    return True
            except Exception:
                continue
        return False
    except Exception as e:
        print(f"{LOG_PREFIX} [DB][FOLDER_CHECK][ERR] {repr(e)}")
        return False

def _db_insert(path, emb):
    try:
        if _cur is None:
            return
        _cur.execute("INSERT INTO faces(ts,path,emb) VALUES(?,?,?)",
                     (time.time(), path, emb.tobytes()))
        _conn.commit()
    except Exception as e:
        print(f"{LOG_PREFIX} [DB][INSERT][ERR] {repr(e)}")

# -------------------------------------------------------------
# Window handle helpers and state for scrcpy capture hardening
# -------------------------------------------------------------
WINDOW_TITLE_SUBSTR = "Foresight Phone Mirror"
SCRCPY_HWND = None
SCRCPY_LAST_OK = 0.0  # timestamp of last successful GetWindowRect
SCRCPY_OK_CONSEC = 0  # count of consecutive successful GetWindowRect calls
_WIN_INVALID_LOG_LAST = 0.0
_WIN_WAIT_LOG_LAST = 0.0
_WIN_GETRECT_ERR_LAST = 0.0

def get_hwnd_by_title_substr(substr):
    """Return first visible HWND whose title contains substr (case-insensitive)."""
    try:
        matches = []
        sub = (substr or "").lower()
        def enum_windows_callback(hwnd, windows):
            try:
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd) or ""
                    if sub in title.lower():
                        windows.append(hwnd)
            except Exception:
                pass
            return True
        win32gui.EnumWindows(enum_windows_callback, matches)
        return matches[0] if matches else None
    except Exception:
        return None

def is_valid_hwnd(hwnd):
    """Return True if hwnd is a valid and visible window."""
    try:
        return bool(hwnd) and bool(win32gui.IsWindow(hwnd)) and bool(win32gui.IsWindowVisible(hwnd))
    except Exception:
        return False

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
    def __init__(self, model_name="yolo11n.pt", use_fallback=False):  # Use nano model for speed
        self.model = None
        self.model_name = model_name
        self.confidence_threshold = 0.15  # Lowered for higher detection rate - catch more partial humans
        # Initialize object tracker for ultra-responsive detection
        self.tracker = ObjectTracker(
            smoothing_factor=0.5,  # Minimal smoothing for maximum responsiveness
            max_distance=80,
            frame_skip_interval=1,   # Process EVERY frame for maximum speed
            grace_period_ms=100     # Ultra-short grace period for instant response
        )
        self.use_fallback = use_fallback
        # ULTRA-HIGH FREQUENCY: No frame skipping - maximum detection speed
        self.detection_counter = 0
        
        # Ultra-responsive detection for real-time performance
        self.previous_detections = []
        self.persistent_detections = {}  # Track detections across frames
        self.smoothing_factor = 0.3  # Minimal smoothing for instant updates
        self.min_confidence_diff = 0.1   # Lower threshold for faster updates
        self.detection_persistence = 5   # Very short persistence for real-time response
        self.min_confidence_for_new = 0.12  # Even lower threshold for maximum detection coverage
        
        # Optimized motion detection for ultra-low latency
        self.prev_frame = None
        self.motion_threshold = 10.0  # Ultra-sensitive motion detection for better coverage
        self.frame_buffer = []  # Buffer to store recent frames
        self.buffer_size = 2  # Smaller buffer for lower latency
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

        # Face save configuration and recent frames buffer
        self.face_save_dir = None
        self.enable_face_save = False
        self.recent_frames = deque(maxlen=15)
        self.last_face_save_time = 0
        self.face_save_cooldown = 5.0  # seconds
        self.face_compare_limit = 20  # max existing images to compare (performance)
        self.face_compare_extensions = {'.jpg', '.jpeg', '.png'}
        # Background face save worker state
        self.save_worker_busy = False
        # DeepFace model cache (prepared asynchronously to avoid blocking detection loop)
        self._df_model = None
        self._df_module = None
        self._df_model_ready = False
        self._df_model_error = None
        try:
            self._prepare_deepface_model_async()
        except Exception as e:
            print(f"[WARNING] Failed to start DeepFace model preparation: {e}")
        
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

    def set_face_save_dir(self, dir_path, enable=True):
        try:
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
                self.face_save_dir = dir_path
                self.enable_face_save = bool(enable)
                print(f"[INFO] Face save directory set: {dir_path} (enabled={self.enable_face_save})")
            else:
                self.face_save_dir = None
                self.enable_face_save = False
        except Exception as e:
            print(f"[ERROR] Failed to set face save dir: {e}")

    def _capture_and_verify_faces(self, bbox):
        """Capture up to 5 crops around the detected person and verify with DeepFace.
        If they are the same person, keep one image and discard the rest.
        Returns saved file path or None.
        """
        try:
            if not self.enable_face_save or not self.face_save_dir:
                return None

            # Use recent frames to get temporal diversity
            frames = list(self.recent_frames)[-5:]
            if not frames:
                return None

            x1, y1, x2, y2 = map(int, bbox)
            crops = []
            for f in frames:
                # Bound check
                h, w = f.shape[:2]
                x1c = max(0, min(w - 1, x1))
                y1c = max(0, min(h - 1, y1))
                x2c = max(0, min(w - 1, x2))
                y2c = max(0, min(h - 1, y2))
                if x2c <= x1c or y2c <= y1c:
                    continue
                crops.append(f[y1c:y2c, x1c:x2c].copy())

            # Prefer 2+ crops for verification, but allow 1 for fallback
            if len(crops) < 1:
                return None

            # Detect and focus on face regions within person crops for better verification
            face_crops = []
            try:
                for c in crops:
                    gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
                    faces = []
                    try:
                        # Use OpenCV Haar cascade if available
                        if hasattr(self, 'face_cascade') and self.face_cascade is not None:
                            faces = self.face_cascade.detectMultiScale(
                                gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
                            )
                    except Exception:
                        faces = []
                    if isinstance(faces, (list, tuple)) and len(faces) > 0:
                        # Choose largest detected face
                        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
                        # Bound check inside crop
                        x = max(0, min(x, c.shape[1] - 1))
                        y = max(0, min(y, c.shape[0] - 1))
                        w = max(1, min(w, c.shape[1] - x))
                        h = max(1, min(h, c.shape[0] - y))
                        face_crops.append(c[y:y + h, x:x + w].copy())
                    else:
                        # Fallback to the whole person crop if no face found
                        face_crops.append(c)
            except Exception:
                # Any error in face detection, fallback to original crops
                face_crops = crops

            # Use prebuilt DeepFace model; skip if not ready to avoid blocking
            if not self._df_model_ready or not self._df_module:
                if self._df_model_error:
                    print(f"[ERROR] DeepFace initialization failed: {self._df_model_error}")
                else:
                    print("[INFO] DeepFace model preparing in background; skipping save until ready")
                return None
            DeepFace = self._df_module

            # Compare first crop to others
            base = face_crops[0]
            same_count = 0
            total_comparisons = max(0, len(face_crops) - 1)
            for i in range(1, len(face_crops)):
                try:
                    # Use API compatible with installed DeepFace: pass model_name instead of model object
                    # Use mediapipe backend for robust face alignment across frames
                    result = DeepFace.verify(base, face_crops[i], model_name='VGG-Face', detector_backend='mediapipe', enforce_detection=False)
                    if result and result.get('verified'):
                        same_count += 1
                except Exception as ve:
                    print(f"[WARNING] DeepFace verify error: {ve}")

            # Use majority threshold instead of near-all to reduce false negatives
            required = max(1, len(crops) // 2)
            print(f"[INFO] DeepFace verification counts: same={same_count}, total={total_comparisons}, required={required}")
            if same_count >= required:
                # Check duplicates against existing saved faces before saving
                try:
                    if self._is_duplicate_face(base):
                        print("[INFO] Duplicate face found in folder; skipping save")
                        return None
                except Exception as dupe_e:
                    print(f"[WARNING] Duplicate check failed: {dupe_e}")
                ts = time.strftime('%Y%m%d_%H%M%S')
                filename = f"face_{ts}_{x1}_{y1}_{x2}_{y2}.jpg"
                save_path = os.path.join(self.face_save_dir, filename)
                try:
                    cv2.imwrite(save_path, base)
                    print(f"FACE_SAVED:{save_path}", flush=True)
                    return save_path
                except Exception as se:
                    print(f"[ERROR] Failed to save face image: {se}")
            else:
                print("[INFO] DeepFace verification did not confirm same person; skipping save")

        except Exception as e:
            print(f"[ERROR] Face capture/verify failed: {e}")
            return None

    def _capture_and_verify_faces_from_frames(self, frames, bbox):
        """Same as _capture_and_verify_faces but uses provided frames snapshot.
        Runs DeepFace verification and duplicate check, saves unique face.
        """
        try:
            if not self.enable_face_save or not self.face_save_dir:
                return None

            if not frames:
                return None

            x1, y1, x2, y2 = map(int, bbox)
            crops = []
            for f in frames:
                h, w = f.shape[:2]
                x1c = max(0, min(w - 1, x1))
                y1c = max(0, min(h - 1, y1))
                x2c = max(0, min(w - 1, x2))
                y2c = max(0, min(h - 1, y2))
                if x2c <= x1c or y2c <= y1c:
                    continue
                crops.append(f[y1c:y2c, x1c:x2c].copy())

            if len(crops) < 1:
                return None

            # Detect and focus on face regions within person crops
            face_crops = []
            try:
                for c in crops:
                    gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
                    faces = []
                    try:
                        if hasattr(self, 'face_cascade') and self.face_cascade is not None:
                            faces = self.face_cascade.detectMultiScale(
                                gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
                            )
                    except Exception:
                        faces = []
                    if isinstance(faces, (list, tuple)) and len(faces) > 0:
                        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
                        x = max(0, min(x, c.shape[1] - 1))
                        y = max(0, min(y, c.shape[0] - 1))
                        w = max(1, min(w, c.shape[1] - x))
                        h = max(1, min(h, c.shape[0] - y))
                        face_crops.append(c[y:y + h, x:x + w].copy())
                    else:
                        face_crops.append(c)
            except Exception:
                face_crops = crops

            if not self._df_model_ready or not self._df_module:
                if self._df_model_error:
                    print(f"[ERROR] DeepFace initialization failed: {self._df_model_error}", flush=True)
                else:
                    print("EVENT_DF_PREPARING", flush=True)
                return None
            DeepFace = self._df_module

            base = face_crops[0]
            same_count = 0
            total_comparisons = max(0, len(face_crops) - 1)
            for i in range(1, len(face_crops)):
                try:
                    result = DeepFace.verify(base, face_crops[i], model_name='VGG-Face', detector_backend='mediapipe', enforce_detection=False)
                    if result and result.get('verified'):
                        same_count += 1
                except Exception as ve:
                    print(f"[WARNING] DeepFace verify error: {ve}", flush=True)

            required = max(1, len(crops) // 2)
            print(f"EVENT_VERIFICATION_COUNTS same={same_count} total={total_comparisons} required={required}", flush=True)
            if same_count >= required:
                try:
                    if self._is_duplicate_face(base):
                        print("EVENT_DUPLICATE_MATCH", flush=True)
                        return None
                except Exception as dupe_e:
                    print(f"[WARNING] Duplicate check failed: {dupe_e}", flush=True)
                ts = time.strftime('%Y%m%d_%H%M%S')
                filename = f"face_{ts}_{x1}_{y1}_{x2}_{y2}.jpg"
                save_path = os.path.join(self.face_save_dir, filename)
                try:
                    cv2.imwrite(save_path, base)
                    print(f"FACE_SAVED:{save_path}", flush=True)
                    return save_path
                except Exception as se:
                    print(f"[ERROR] Failed to save face image: {se}", flush=True)
            else:
                print("EVENT_VERIFICATION_REJECT", flush=True)

        except Exception as e:
            print(f"[ERROR] Face capture/verify (frames) failed: {e}", flush=True)
            return None

    def _enqueue_face_save_job(self, frames, bbox):
        """Spawn a background worker to process face verification and saving."""
        try:
            if self.save_worker_busy:
                return False
            if not self.enable_face_save or not self.face_save_dir:
                return False
            if not frames or len(frames) < 5:
                print(f"EVENT_FACE_JOB_SKIPPED_INSUFFICIENT_FRAMES count={len(frames) if frames else 0}", flush=True)
                return False
            self.save_worker_busy = True
            import threading
            print(f"EVENT_FACE_JOB_ENQUEUED frames={len(frames)} bbox={bbox}", flush=True)
            def _worker():
                try:
                    print("EVENT_FACE_JOB_START", flush=True)
                    saved = self._capture_and_verify_faces_from_frames(frames, bbox)
                    if saved:
                        print(f"EVENT_FACE_JOB_DONE saved={saved}", flush=True)
                    else:
                        print("EVENT_FACE_JOB_DONE saved=None", flush=True)
                finally:
                    self.save_worker_busy = False
            t = threading.Thread(target=_worker, name="face-save-worker", daemon=True)
            t.start()
            return True
        except Exception as e:
            self.save_worker_busy = False
            print(f"[ERROR] Failed to enqueue face save job: {e}", flush=True)
            return False

    # -------------------------------------------------------------
    # Foresight per-track 5-frame pipeline (IoU/centroid tracker)
    # -------------------------------------------------------------
    def _init_foresight_tracking(self):
        if not hasattr(self, 'fs_tracks'):
            self.fs_tracks = {}
            self.fs_frames_seen = {}
            self.fs_burst_buf = {}
            self.fs_next_track_id = 1
            self.fs_last_crop_ts = {}

    def _bbox_from_detection(self, d):
        try:
            if 'coords' in d:
                x1, y1, x2, y2 = map(int, d['coords'])
                return (x1, y1, x2, y2)
            else:
                x1 = int(d.get('x1', 0)); y1 = int(d.get('y1', 0))
                x2 = int(d.get('x2', 0)); y2 = int(d.get('y2', 0))
                return (x1, y1, x2, y2)
        except Exception:
            return None

    def _is_person_detection(self, d):
        try:
            if 'coords' in d:
                cls_id = int(d.get('cls', 0))
                label = self.get_class_name(cls_id)
                return (label or '').lower() == 'person'
            else:
                label = (d.get('label') or '').lower()
                return label == 'person'
        except Exception:
            return False

    def _iou(self, b1, b2):
        try:
            x1, y1, x2, y2 = b1
            x1b, y1b, x2b, y2b = b2
            ix1 = max(x1, x1b); iy1 = max(y1, y1b)
            ix2 = min(x2, x2b); iy2 = min(y2, y2b)
            iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
            inter = iw * ih
            area1 = max(0, x2 - x1) * max(0, y2 - y1)
            area2 = max(0, x2b - x1b) * max(0, y2b - y1b)
            union = area1 + area2 - inter
            if union <= 0:
                return 0.0
            return inter / union
        except Exception:
            return 0.0

    def _centroid(self, b):
        x1, y1, x2, y2 = b
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _assign_or_create_track(self, bbox):
        self._init_foresight_tracking()
        best_tid, best_iou = None, 0.0
        for tid, t in self.fs_tracks.items():
            i = self._iou(bbox, t['bbox'])
            if i > best_iou:
                best_tid, best_iou = tid, i
        if best_tid is not None and best_iou >= 0.3:
            self.fs_tracks[best_tid]['bbox'] = bbox
            return best_tid, bbox
        cx, cy = self._centroid(bbox)
        best_tid, best_dist = None, 1e9
        for tid, t in self.fs_tracks.items():
            tcx, tcy = self._centroid(t['bbox'])
            dist = (tcx - cx) ** 2 + (tcy - cy) ** 2
            if dist < best_dist:
                best_tid, best_dist = tid, dist
        if best_tid is not None and best_dist <= (50 ** 2):
            self.fs_tracks[best_tid]['bbox'] = bbox
            return best_tid, bbox
        tid = self.fs_next_track_id
        self.fs_next_track_id += 1
        self.fs_tracks[tid] = { 'bbox': bbox, 'last_seen': time.time() }
        self.fs_frames_seen[tid] = 0
        self.fs_burst_buf.setdefault(tid, [])
        return tid, bbox

    def _on_detection(self, tid, frame, bbox):
        self._init_foresight_tracking()
        c = self.fs_frames_seen.get(tid, 0) + 1
        self.fs_frames_seen[tid] = c
        buf = self.fs_burst_buf.setdefault(tid, [])
        buf.append((frame.copy(), bbox))
        # Diagnostic trace after counting frames
        print(f"{LOG_PREFIX} [TRACE][{tid}] frames={c}", flush=True)

        # Save cropped detection every CROP_SAVE_INTERVAL_S seconds per track
        try:
            now = time.time()
            last = self.fs_last_crop_ts.get(tid, 0)
            if now - last >= CROP_SAVE_INTERVAL_S:
                x1, y1, x2, y2 = map(int, bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                if x2 > x1 and y2 > y1:
                    # Include green bounded box inside cropped image
                    crop = frame[y1:y2, x1:x2].copy()
                    try:
                        import cv2
                        cv2.rectangle(crop, (1, 1), (max(2, crop.shape[1]-2), max(2, crop.shape[0]-2)), (0, 255, 0), 2)
                    except Exception:
                        pass
                    fname = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_t{tid}_crop.jpg"
                    outp = os.path.join(SAVE_DIR, fname)
                    try:
                        ok = cv2.imwrite(outp, crop)
                        print(f"{LOG_PREFIX} [CROP][SAVE] id={tid} -> {outp} ok={ok}")
                    except Exception as e:
                        print(f"{LOG_PREFIX} [CROP][ERR] id={tid} {e}")
                self.fs_last_crop_ts[tid] = now
        except Exception as e:
            print(f"{LOG_PREFIX} [CROP][ERR] {e}")
        if c >= FRAMES_REQUIRED:
            def _job():
                try:
                    self._run_face_job(tid, buf[:FRAMES_REQUIRED])
                except Exception as e:
                    print(f"{LOG_PREFIX}[FACE_JOB][{tid}]ERR:{e!r}", flush=True)
                finally:
                    try:
                        self.fs_burst_buf[tid].clear()
                        self.fs_frames_seen[tid] = 0
                    except Exception:
                        pass
            try:
                import threading
                threading.Thread(target=_job, name=f"fs-face-job-{tid}", daemon=True).start()
            except Exception as e:
                print(f"{LOG_PREFIX}[FACE_JOB][{tid}]thread_err:{e!r}", flush=True)

    # Minimal saver: no ML, no dedupe; crop and save per track with cooldown
    def on_detection(self, track_id, frame, bbox):
        try:
            now = time.time()
            last = track_last_save.get(track_id, 0)
            if now - last >= PER_TRACK_COOLDOWN_S:
                x1, y1, x2, y2 = map(int, bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                crop = frame[y1:y2, x1:x2].copy()
                try:
                    import cv2
                    cv2.rectangle(crop, (1, 1), (max(2, crop.shape[1]-2), max(2, crop.shape[0]-2)), (0, 255, 0), 2)
                except Exception:
                    pass
                if crop.size and crop.shape[0] >= 40 and crop.shape[1] >= 40:
                    fname = f"{int(now)}_t{track_id}.jpg"
                    outp = os.path.join(SAVE_DIR, fname)
                    ok = cv2.imwrite(outp, crop)
                    print(f"[FORESIGHT][SAVE] {outp} ok={ok}")
                    track_last_save[track_id] = now
        except Exception as e:
            print(f"{LOG_PREFIX} [on_detection ERROR] {e}", flush=True)

    def _run_face_job(self, tid, frames_burst):
        print(f"{LOG_PREFIX}[FACE_JOB][{tid}]start;burst={len(frames_burst)}", flush=True)
        # Non-blocking loading indicator
        try:
            show_loading(True)
        except Exception as _e:
            print(f"{LOG_PREFIX}[UI] show_loading_err:{_e}", flush=True)
        faces = []
        for i, (f, b) in enumerate(frames_burst):
            try:
                x1, y1, x2, y2 = map(int, b)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(f.shape[1], x2), min(f.shape[0], y2)
                crop = f[y1:y2, x1:x2]
                if crop.shape[0] < MIN_FACE_SIZE or crop.shape[1] < MIN_FACE_SIZE:
                    continue
                try:
                    from deepface import DeepFace
                    os.makedirs(r"C:\Users\Asus\.deepface\weights", exist_ok=True)
                    print("[FORESIGHT][DeepFace] Running comparison...")
                    fc = DeepFace.extract_faces(img_path=crop, detector_backend=FACE_DETECTOR_BACKEND, enforce_detection=False)
                    if fc:
                        faces.append((fc[0]["face"] * 255).astype("uint8"))
                except Exception as e:
                    msg = str(e)
                    if "arcface_weights.h5" in msg.lower():
                        print("[FORESIGHT][DeepFace] Missing weights. Place file at C:\\Users\\Asus\\.deepface\\weights\\arcface_weights.h5", flush=True)
                    print(f"{LOG_PREFIX}[FACE_JOB][{tid}]extracterr:{repr(e)}", flush=True)
            except Exception:
                continue
        if not faces:
            print(f"{LOG_PREFIX}[FACE_JOB][{tid}]no_faces", flush=True)
            try:
                show_loading(False)
            except Exception as _e:
                print(f"{LOG_PREFIX}[UI] hide_loading_err:{_e}", flush=True)
            return
        rep = faces[min(2, len(faces) - 1)]
        if FORCE_ALWAYS_SAVE or BYPASS_DEEPFACE:
            self._save_rep(tid, rep, "BYPASS" if BYPASS_DEEPFACE else "FORCE")
            try:
                show_loading(False)
            except Exception as _e:
                print(f"{LOG_PREFIX}[UI] hide_loading_err:{_e}", flush=True)
            return
        seen = self._check_in_db(rep)
        print(f"{LOG_PREFIX}[FACE_JOB][{tid}]seen?{seen}", flush=True)
        if not seen:
            self._save_rep(tid, rep, "NEW")
        else:
            print(f"{LOG_PREFIX}[FACE_JOB][{tid}]skip(existing)", flush=True)
        try:
            show_loading(False)
        except Exception as _e:
            print(f"{LOG_PREFIX}[UI] hide_loading_err:{_e}", flush=True)

    def _save_rep(self, tid, img, tag="NEW"):
        fn = f"{int(time.time())}_{tid}_{uuid.uuid4().hex}.jpg"
        p = os.path.join(SAVE_DIR, fn)
        try:
            ok = cv2.imwrite(p, img)
        except Exception:
            ok = False
        print(f"{LOG_PREFIX}[SAVE][{tag}][{tid}]>{p}ok={ok}", flush=True)
        if ok:
            try:
                cv2.imwrite(os.path.join(DB_DIR, fn), img)
                print(f"FACE_SAVED:{p}", flush=True)
            except Exception as e:
                print(f"{LOG_PREFIX}[DB_SAVE][{tid}]err:{e}", flush=True)

    def _check_in_db(self, img):
        tmp = os.path.join(SAVE_DIR, "_tmp.jpg")
        try:
            cv2.imwrite(tmp, img)
            from deepface import DeepFace
            os.makedirs(r"C:\Users\Asus\.deepface\weights", exist_ok=True)
            print("[FORESIGHT][DeepFace] Running comparison...")
            try:
                r = DeepFace.find(img_path=tmp, db_path=DB_DIR, model_name=DEEPFACE_MODEL,
                                  detector_backend=FACE_DETECTOR_BACKEND, distance_metric=DEEPFACE_METRIC,
                                  enforce_detection=False, silent=True)
            except Exception as e:
                msg = str(e)
                if "arcface_weights.h5" in msg.lower():
                    print("[FORESIGHT][DeepFace] Missing weights. Place file at C:\\Users\\Asus\\.deepface\\weights\\arcface_weights.h5", flush=True)
                print(f"{LOG_PREFIX}[DB]err:{repr(e)}", flush=True)
                return False
            if not r or getattr(r[0], 'empty', False):
                return False
            df = r[0]
            col = None
            for c in df.columns:
                if str(c).endswith(DEEPFACE_METRIC):
                    col = c; break
            if not col:
                return False
            try:
                d = float(df.iloc[0][col])
                print(f"{LOG_PREFIX}[DB]dist={d:.3f}thr={DEEPFACE_THRESH}", flush=True)
                return d <= DEEPFACE_THRESH
            except Exception:
                return False
        except Exception as e:
            print(f"{LOG_PREFIX}[DB]err:{e}", flush=True)
            return False
        finally:
            try:
                os.remove(tmp)
            except Exception:
                pass

    def _is_duplicate_face(self, candidate_img):
        """Compare candidate image to existing images in face_save_dir using DeepFace.
        Returns True if any verified match is found, False otherwise.
        """
        try:
            if not self.face_save_dir:
                return False
            if not self._df_model_ready or not self._df_module:
                # If DeepFace is not ready yet, skip duplicate check (avoid blocking)
                return False
            DeepFace = self._df_module
            # Collect existing image paths
            files = []
            for name in os.listdir(self.face_save_dir):
                ext = os.path.splitext(name)[1].lower()
                if ext in self.face_compare_extensions:
                    files.append(os.path.join(self.face_save_dir, name))
            if not files:
                return False
            # Sort by modification time desc and limit
            files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            files = files[: self.face_compare_limit]
            matches = 0
            compared = 0
            for path in files:
                try:
                    img = cv2.imread(path)
                    if img is None:
                        continue
                    compared += 1
                    # Use API compatible with installed DeepFace
                    # Use mediapipe backend for folder comparisons as well
                    res = DeepFace.verify(candidate_img, img, model_name='VGG-Face', detector_backend='mediapipe', enforce_detection=False)
                    if res and res.get('verified'):
                        matches += 1
                        # One match is enough to mark as duplicate
                        print(f"[INFO] Folder duplicate matched: {os.path.basename(path)}")
                        print(f"[INFO] DeepFace folder compare: compared={compared}, matches={matches}")
                        return True
                except Exception as e:
                    # Ignore per-file errors, continue
                    pass
            print(f"[INFO] DeepFace folder compare: compared={compared}, matches={matches}")
            return False
        except Exception as e:
            print(f"[WARNING] Duplicate check error: {e}")
            return False

    def _prepare_deepface_model_async(self):
        """Prepare the DeepFace model in a background thread to avoid blocking detection."""
        import threading

        def _worker():
            try:
                print("[INFO] Preparing DeepFace model in background (first run may download ~580MB)")
                # Patch TensorFlow __version__ if missing before importing DeepFace
                try:
                    import tensorflow as tf
                    if not hasattr(tf, "__version__") or tf.__version__ is None:
                        try:
                            import importlib.metadata as im
                            ver = None
                            for d in ("tensorflow", "tensorflow-intel"):
                                try:
                                    ver = im.version(d)
                                    if ver:
                                        break
                                except Exception:
                                    pass
                            setattr(tf, "__version__", ver or "2.15.1")
                        except Exception:
                            setattr(tf, "__version__", "2")
                except Exception:
                    pass

                from deepface import DeepFace
                # Build and cache the model once
                model = DeepFace.build_model("VGG-Face")
                self._df_model = model
                self._df_module = DeepFace
                self._df_model_ready = True
                print("[INFO] DeepFace model ready", flush=True)
            except Exception as e:
                self._df_model_error = e
                self._df_model_ready = False
                print(f"[ERROR] DeepFace model preparation failed: {e}")

        t = threading.Thread(target=_worker, name="deepface-prep", daemon=True)
        t.start()
    
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
            
            # Enhanced multi-scale detection for improved detection rate
            height, width = frame.shape[:2]
            
            # Multi-scale detection: try different resolutions for better coverage
            detection_scales = [640, 416]  # Multiple scales for better detection
            all_detections = []
            
            for target_size in detection_scales:
                if width > target_size or height > target_size:
                    if width > height:
                        scale = target_size / width
                        new_width = target_size
                        new_height = int(height * scale)
                    else:
                        scale = target_size / height
                        new_height = target_size
                        new_width = int(width * scale)
                    frame_resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                else:
                    frame_resized = frame
                    scale = 1.0
                
                # Enhanced preprocessing for better detection
                # Apply histogram equalization for better contrast
                frame_enhanced = cv2.convertScaleAbs(frame_resized, alpha=1.1, beta=10)
                
                # Use the enhanced frame for detection
                frame_to_detect = frame_enhanced
            
                # Enhanced detection with improved NMS settings
                try:
                    if self.device == "cuda":
                        # Enhanced GPU tracking with lower IoU for better detection coverage
                        results = self.model.track(
                            frame_to_detect,
                            persist=True,
                            tracker='bytetrack.yaml',
                            conf=self.confidence_threshold,
                            iou=0.3,  # Lower IoU for better detection of partial/overlapping humans
                            verbose=False,
                            imgsz=target_size,
                            half=True,
                            device=self.device,
                            max_det=100,  # Increased for better coverage
                            classes=[0],  # Only detect persons
                            agnostic_nms=True,
                            augment=True  # Enable test-time augmentation for better detection
                        )
                    else:
                        results = self.model.track(
                            frame_to_detect,
                            persist=True,
                            tracker='bytetrack.yaml',
                            conf=self.confidence_threshold,
                            iou=0.3,  # Lower IoU for better detection coverage
                            verbose=False,
                            imgsz=target_size,
                            max_det=100,
                            classes=[0],
                            agnostic_nms=True,
                            augment=True
                        )
                except:
                    # Enhanced fallback detection
                    if self.device == "cuda":
                        results = self.model(frame_to_detect, conf=self.confidence_threshold, verbose=False, show=False, imgsz=target_size, half=True, device=self.device, max_det=100, classes=[0], agnostic_nms=True, augment=True)
                    else:
                        results = self.model(frame_to_detect, conf=self.confidence_threshold, verbose=False, show=False, imgsz=target_size, max_det=100, classes=[0], agnostic_nms=True, augment=True)
            
                scale_detections = []
                # Process results from current scale
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
                            
                            # Get class name from YOLO's built-in names
                            class_name = self.model.names[class_id] if class_id in self.model.names else f'class_{class_id}'
                            
                            detection = {
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(confidence),
                                'class_id': class_id,
                                'label': class_name,
                                'track_id': track_id,
                                'scale': target_size
                            }
                            scale_detections.append(detection)
                
                all_detections.extend(scale_detections)
            
            # Merge and deduplicate detections from multiple scales
            detections = self._merge_multi_scale_detections(all_detections)
            
            self.last_detections = detections
            self.last_inference_time = current_time
            return detections
            
        except Exception as e:
            print(f"[ERROR] Detection failed: {e}")
            return getattr(self, 'last_detections', [])
    
    def _merge_multi_scale_detections(self, all_detections):
        """Merge detections from multiple scales, removing duplicates and keeping best confidence"""
        if not all_detections:
            return []
        
        # Group detections by overlap (IoU > 0.5 means same object)
        merged_detections = []
        used_indices = set()
        
        for i, det1 in enumerate(all_detections):
            if i in used_indices:
                continue
                
            # Find all overlapping detections
            overlapping = [det1]
            used_indices.add(i)
            
            for j, det2 in enumerate(all_detections[i+1:], i+1):
                if j in used_indices:
                    continue
                    
                # Calculate IoU between bounding boxes
                iou = self._calculate_iou(det1['bbox'], det2['bbox'])
                if iou > 0.5:  # Same object detected at different scales
                    overlapping.append(det2)
                    used_indices.add(j)
            
            # Keep detection with highest confidence
            best_detection = max(overlapping, key=lambda x: x['confidence'])
            merged_detections.append(best_detection)
        
        return merged_detections
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
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
            # Maintain recent frames buffer
            try:
                self.recent_frames.append(frame.copy())
            except Exception:
                pass
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

                        # Legacy cooldown-based face save flow disabled; using per-track 5-frame pipeline
                        # if False:
                        #     now = time.time()
                        #     if self.enable_face_save and (now - self.last_face_save_time) >= self.face_save_cooldown:
                        #         first = detection_data[0]
                        #         bbox = (first['x1'], first['y1'], first['x2'], first['y2'])
                        #         frames_snapshot = list(self.recent_frames)[-5:]
                        #         if self._enqueue_face_save_job(frames_snapshot, bbox):
                        #             self.last_face_save_time = now
                    
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

                # Burst capture hook (person only)
                try:
                    if label == 'person':
                        # derive bbox
                        if 'bbox' in detection:
                            x1, y1, x2, y2 = detection['bbox']
                        else:
                            x1, y1, x2, y2 = x, y, x + w, y + h
                        track_id = detection.get('track_id')
                        if track_id is not None:
                            now = time.time()
                            global burst_buf, burst_started
                            if 'burst_buf' not in globals():
                                burst_buf = {}
                            if 'burst_started' not in globals():
                                burst_started = {}
                            if track_id not in burst_started:
                                burst_started[track_id] = now
                            buf = burst_buf.setdefault(track_id, [])
                            if len(buf) < FRAMES_REQUIRED and (now - burst_started[track_id]) <= CAPTURE_TIMEOUT_S:
                                x1i,y1i,x2i,y2i = map(int, (x1,y1,x2,y2))
                                x1i,y1i = max(0,x1i), max(0,y1i)
                                x2i,y2i = min(frame.shape[1],x2i), min(frame.shape[0],y2i)
                                crop = frame[y1i:y2i, x1i:x2i].copy()
                                if crop.shape[0] >= MIN_FACE_WH and crop.shape[1] >= MIN_FACE_WH:
                                    buf.append((crop, now))
                                    print(f"{LOG_PREFIX} [BURST] id={track_id} n={len(buf)}")
                            # Process immediately when exactly FRAMES_REQUIRED frames are captured
                            if track_id in burst_started and len(buf) >= FRAMES_REQUIRED:
                                try:
                                    _process_burst(track_id)
                                except Exception as e:
                                    print(f"{LOG_PREFIX} [BURST][ERR] id={track_id} {e}")
                                finally:
                                    burst_buf[track_id] = []
                                    burst_started.pop(track_id, None)
                            # Cleanup stale buffers on timeout
                            elif track_id in burst_started and (now - burst_started[track_id]) > CAPTURE_TIMEOUT_S:
                                burst_buf[track_id] = []
                                burst_started.pop(track_id, None)
                except Exception:
                    pass
            
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
        # Performance optimizations
        self.cached_monitor = None
        self.frame_buffer = None
        self.last_dimensions = None
        
    def capture_screen(self):
        """Optimized screen capture with caching for real-time performance"""
        try:
            import mss
            
            # Cache monitor configuration for better performance
            if self.cached_monitor is None:
                if self.region:
                    x, y, width, height = self.region
                    self.cached_monitor = {"top": y, "left": x, "width": width, "height": height}
                else:
                    with mss.mss() as sct:
                        self.cached_monitor = sct.monitors[1]  # Primary monitor
            
            with mss.mss() as sct:
                screenshot = sct.grab(self.cached_monitor)
                
                # Get current dimensions
                current_dims = (screenshot.height, screenshot.width)
                
                # Pre-allocate buffer if needed
                if self.last_dimensions != current_dims or self.frame_buffer is None:
                    self.frame_buffer = np.empty((screenshot.height, screenshot.width, 4), dtype='uint8')
                    self.last_dimensions = current_dims
                
                # Convert screenshot data to numpy array
                frame_data = np.frombuffer(screenshot.bgra, dtype='uint8')
                self.frame_buffer = frame_data.reshape((screenshot.height, screenshot.width, 4))
                
                # Optimized color conversion
                frame = cv2.cvtColor(self.frame_buffer, cv2.COLOR_BGRA2BGR)
                
                return frame
        except ImportError:
            print("[ERROR] mss library not found. Install with: pip install mss")
            return None
        except Exception as e:
            print(f"[ERROR] Screen capture failed: {e}")
            return None

class WindowOverlay:
    def __init__(self, window_title="Foresight Phone Mirror"):
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

    def draw_loading_message(self, message="Analyzing faces..."):
        """Draw a bottom-centered loading message on the target window."""
        if not self.hwnd:
            return
        try:
            import win32gui
            import win32con
            import win32api
            # Acquire DC and set transparent background
            hdc = win32gui.GetWindowDC(self.hwnd)
            win32gui.SetBkMode(hdc, win32con.TRANSPARENT)
            win32gui.SetTextColor(hdc, win32api.RGB(255, 255, 0))  # Yellow text
            # Compute bottom-center rect
            left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
            width = right - left
            height = bottom - top
            rect = (int(width * 0.25), int(height - 40), int(width * 0.75), int(height - 10))
            # Centered single line text
            flags = win32con.DT_CENTER | win32con.DT_SINGLELINE
            win32gui.DrawText(hdc, message, -1, rect, flags)
            win32gui.ReleaseDC(self.hwnd, hdc)
        except Exception as e:
            print(f"[ERROR] Failed to draw loading message: {e}")

    def clear_loading_message(self):
        """Request the target window to redraw, clearing overlay text."""
        if not self.hwnd:
            return
        try:
            import win32gui
            import win32con
            win32gui.RedrawWindow(
                self.hwnd,
                None,
                None,
                win32con.RDW_INVALIDATE | win32con.RDW_ERASE | win32con.RDW_ALLCHILDREN,
            )
        except Exception as e:
            print(f"[ERROR] Failed to clear loading message: {e}")
    
    def cleanup(self):
        """Clean up overlay window"""
        if self.overlay_hwnd:
            try:
                win32gui.DestroyWindow(self.overlay_hwnd)
            except:
                pass
            self.overlay_hwnd = None

def _process_burst(track_id):
    items = burst_buf.get(track_id, [])
    if not items:
        return
    items = sorted(items, key=lambda x: x[1])[:FRAMES_REQUIRED]
    crops = [c for (c,_) in items]

    embs = []
    for c in crops:
        e = _embed_one(c)
        if e is not None:
            embs.append(e)
    if len(embs) < 3:
        print(f"{LOG_PREFIX} [BURST] id={track_id} insufficient faces -> skip")
        return

    # same-person: all pairwise <= SAME_PERSON_THRESH
    ok = True
    for i in range(len(embs)):
        for j in range(i+1, len(embs)):
            if _cosine_dist(embs[i], embs[j]) > SAME_PERSON_THRESH:
                ok = False
                break
        if not ok:
            break
    if not ok:
        print(f"{LOG_PREFIX} [BURST] id={track_id} not same person -> skip")
        return

    rep_idx = min(2, len(crops)-1)  # middle
    rep_crop = crops[rep_idx]
    rep_emb  = embs[min(rep_idx, len(embs)-1)]

    if _db_any_match(rep_emb, DB_MATCH_THRESH):
        print(f"{LOG_PREFIX} [SAVE] id={track_id} duplicate -> skip")
        return

    # Check within the specified folder (SAVE_DIR) for any similarities
    if _db_any_match_in_folder(rep_emb, DB_MATCH_THRESH):
        print(f"{LOG_PREFIX} [SAVE] id={track_id} folder-duplicate -> skip")
        return

    fname = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_t{track_id}.jpg"
    outp  = os.path.join(SAVE_DIR, fname)
    ok = cv2.imwrite(outp, rep_crop)
    print(f"{LOG_PREFIX} [SAVE] id={track_id} -> {outp} ok={ok}")
    if ok:
        _db_insert(outp, rep_emb)

class WindowCapture:
    def __init__(self, window_title="Foresight Phone Mirror"):
        self.window_title = window_title
        self.hwnd = None
        # Performance optimizations
        self.cached_dc = None
        self.cached_bitmap = None
        self.cached_dimensions = None
        self.frame_buffer = None
        
    def find_window(self):
        """Find window by title"""
        try:
            global SCRCPY_HWND
            hwnd = get_hwnd_by_title_substr(self.window_title)
            if hwnd and is_valid_hwnd(hwnd):
                SCRCPY_HWND = hwnd
                self.hwnd = hwnd
                try:
                    title = win32gui.GetWindowText(hwnd)
                    print(f"[INFO] Found window: {title}")
                except Exception:
                    print("[INFO] Found window")
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
        """Optimized window capture with caching for real-time performance"""
        global SCRCPY_HWND, SCRCPY_LAST_OK, SCRCPY_OK_CONSEC
        global _WIN_INVALID_LOG_LAST, _WIN_WAIT_LOG_LAST, _WIN_GETRECT_ERR_LAST
        # Ensure we have a valid hwnd; handle first-capture and reacquisition paths
        first_boot = SCRCPY_LAST_OK == 0.0
        if not SCRCPY_HWND or not is_valid_hwnd(SCRCPY_HWND):
            now = time.time()
            if now - _WIN_INVALID_LOG_LAST > 2.0:
                print("[FORESIGHT][WIN] handle invalid; re-acquiring…")
                _WIN_INVALID_LOG_LAST = now
            max_attempts = 30 if first_boot else 50
            for _ in range(max_attempts):
                hwnd = get_hwnd_by_title_substr(WINDOW_TITLE_SUBSTR)
                if hwnd and is_valid_hwnd(hwnd):
                    try:
                        SCRCPY_HWND = hwnd
                        self.hwnd = hwnd
                        if not first_boot:
                            print("[FORESIGHT][WIN] reacquired handle")
                        if first_boot:
                            time.sleep(0.5)
                        # Confirm rect once before proceeding
                        left, top, right, bottom = win32gui.GetWindowRect(SCRCPY_HWND)
                        SCRCPY_LAST_OK = time.time()
                        SCRCPY_OK_CONSEC = 1
                        break
                    except Exception:
                        SCRCPY_HWND = None
                        self.hwnd = None
                else:
                    t2 = time.time()
                    if t2 - _WIN_WAIT_LOG_LAST > 3.0:
                        print("[FORESIGHT][WIN] still waiting for scrcpy window…")
                        _WIN_WAIT_LOG_LAST = t2
                time.sleep(0.2)
            if not SCRCPY_HWND:
                return None
        else:
            # Keep self.hwnd in sync
            self.hwnd = SCRCPY_HWND
        
        try:
            import win32gui
            import win32ui
            import win32con
            
            # Get window dimensions
            try:
                left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
                SCRCPY_LAST_OK = time.time()
                SCRCPY_OK_CONSEC = min(SCRCPY_OK_CONSEC + 1, 3)
            except Exception as e:
                now = time.time()
                if now - _WIN_GETRECT_ERR_LAST > 2.5:
                    print(f"[FORESIGHT][WIN] GetWindowRect error: {repr(e)}")
                    _WIN_GETRECT_ERR_LAST = now
                # Invalidate and skip this frame; reacquire next time
                SCRCPY_HWND = None
                self.hwnd = None
                return None
            width = right - left
            height = bottom - top
            current_dimensions = (width, height)
            
            # Check if we need to recreate cached resources
            if (self.cached_dimensions != current_dimensions or 
                self.cached_dc is None or self.cached_bitmap is None):
                
                # Clean up old resources
                self._cleanup_cache()
                
                # Create new cached resources
                hwndDC = win32gui.GetWindowDC(self.hwnd)
                self.mfcDC = win32ui.CreateDCFromHandle(hwndDC)
                self.cached_dc = self.mfcDC.CreateCompatibleDC()
                
                self.cached_bitmap = win32ui.CreateBitmap()
                self.cached_bitmap.CreateCompatibleBitmap(self.mfcDC, width, height)
                self.cached_dc.SelectObject(self.cached_bitmap)
                
                self.cached_dimensions = current_dimensions
                self.hwndDC = hwndDC
                
                # Pre-allocate frame buffer for better performance
                self.frame_buffer = np.empty((height, width, 4), dtype='uint8')
            
            # Fast capture using cached resources
            self.cached_dc.BitBlt((0, 0), (width, height), self.mfcDC, (0, 0), win32con.SRCCOPY)
            
            # Optimized bitmap data extraction
            bmpstr = self.cached_bitmap.GetBitmapBits(True)
            
            # Convert bitmap data to numpy array
            frame_data = np.frombuffer(bmpstr, dtype='uint8')
            self.frame_buffer = frame_data.reshape((height, width, 4))
            
            # Fast color conversion
            frame = cv2.cvtColor(self.frame_buffer, cv2.COLOR_BGRA2BGR)
            
            return frame
            
        except ImportError:
            print("[ERROR] pywin32 library not found. Install with: pip install pywin32")
            return None
        except Exception as e:
            print(f"[ERROR] Window capture failed: {e}")
            self._cleanup_cache()  # Clean up on error
            return None
    
    def _cleanup_cache(self):
        """Clean up cached resources"""
        try:
            if hasattr(self, 'cached_bitmap') and self.cached_bitmap:
                win32gui.DeleteObject(self.cached_bitmap.GetHandle())
            if hasattr(self, 'cached_dc') and self.cached_dc:
                self.cached_dc.DeleteDC()
            if hasattr(self, 'mfcDC') and self.mfcDC:
                self.mfcDC.DeleteDC()
            if hasattr(self, 'hwndDC') and self.hwndDC:
                win32gui.ReleaseDC(self.hwnd, self.hwndDC)
        except:
            pass  # Ignore cleanup errors
        
        self.cached_dc = None
        self.cached_bitmap = None
        self.cached_dimensions = None
    
    def __del__(self):
        """Destructor to clean up resources"""
        self._cleanup_cache()

def show_loading(active: bool, message: str = "Analyzing faces..."):
    """Helper to show or hide a non-blocking loading indicator.
    Uses overlay if available, otherwise prints to console.
    """
    try:
        if active:
            print(f"{LOG_PREFIX} [UI] {message}", flush=True)
            if GLOBAL_WINDOW_OVERLAY:
                try:
                    GLOBAL_WINDOW_OVERLAY.draw_loading_message(message)
                except Exception as e:
                    print(f"{LOG_PREFIX} [UI] draw_loading_err: {e}", flush=True)
        else:
            print(f"{LOG_PREFIX} [UI] Loading complete", flush=True)
            if GLOBAL_WINDOW_OVERLAY:
                try:
                    GLOBAL_WINDOW_OVERLAY.clear_loading_message()
                except Exception as e:
                    print(f"{LOG_PREFIX} [UI] clear_loading_err: {e}", flush=True)
    except Exception as e:
        print(f"{LOG_PREFIX} [UI] show_loading_err: {e}", flush=True)

def main():
    global GLOBAL_WINDOW_OVERLAY
    parser = argparse.ArgumentParser(description='Foresight YOLO Detection')
    parser.add_argument('--source', default='screen', help='Source: screen, camera, window, or file path')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--region', default=None, help='Screen region: x,y,width,height')
    parser.add_argument('--window-title', default='Foresight Phone Mirror', help='Window title to capture')
    parser.add_argument('--output', default=None, help='Output video file path')
    parser.add_argument('--show', action='store_true', help='Display detection window')
    parser.add_argument('--yolo-scrcpy-mode', action='store_true', help='YOLO scrcpy window mode')
    parser.add_argument('--overlay', action='store_true', help='Draw detections directly on target window')
    parser.add_argument('--face-save-dir', default=None, help='Directory to save verified face crops')
    parser.add_argument('--enable-face-save', action='store_true', help='Enable capturing 5 crops and DeepFace verification')
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
    if args.face_save_dir and args.enable_face_save:
        detector.set_face_save_dir(args.face_save_dir, enable=True)
    
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

        # Wait for the target window to appear (scrcpy may take a moment)
        wait_start = time.time()
        wait_timeout = 30.0  # seconds
        while not window_capture.find_window():
            if time.time() - wait_start > wait_timeout:
                print(f"[WARNING] Window '{args.window_title}' not found within {int(wait_timeout)}s; waiting aborted")
                break
            print(f"[INFO] Waiting for '{args.window_title}' window...")
            time.sleep(0.5)

        # Initialize overlay if requested (after window is confirmed)
        window_overlay = None
        if args.overlay:
            window_overlay = WindowOverlay(args.window_title)
            if window_overlay.find_window():
                print(f"[INFO] Overlay mode enabled for '{args.window_title}'")
                # Store global overlay reference for non-blocking loading indicator
                GLOBAL_WINDOW_OVERLAY = window_overlay
            else:
                print(f"[WARNING] Could not find window for overlay mode")
                window_overlay = None
                GLOBAL_WINDOW_OVERLAY = None
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
                    # If capture fails (window not found or invalid handle), retry instead of exiting
                    time.sleep(0.2)
                    continue
            else:  # Camera/video capture
                ret, frame = cap.read()
                if not ret:
                    break
            
            # ULTRA-HIGH FREQUENCY: No frame skipping - detect EVERY millisecond
            # Get stabilized detections and annotated frame - IMMEDIATE PROCESSING
            frame_with_detections, stabilized_detections = detector.detect_and_draw_simple_with_data(frame.copy())

            # Draw person rectangles with cv2 directly before showing the frame
            try:
                if stabilized_detections:
                    for d in stabilized_detections:
                        try:
                            # Determine format and label
                            if 'coords' in d:
                                x1, y1, x2, y2 = map(int, d['coords'])
                                label = detector.get_class_name(d.get('cls', 0))
                            else:
                                x1 = int(d.get('x1', 0)); y1 = int(d.get('y1', 0))
                                x2 = int(d.get('x2', 0)); y2 = int(d.get('y2', 0))
                                label = d.get('class', 'person')
                            if label == 'person':
                                cv2.rectangle(frame_with_detections, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        except Exception:
                            # Skip malformed detection entries
                            continue
            except Exception:
                pass

            # For overlay mode, refresh overlay immediately with stabilized detections
            if window_overlay and stabilized_detections:
                try:
                    overlay_detections = []
                    for d in stabilized_detections:
                        if 'coords' in d:
                            x1, y1, x2, y2 = map(int, d['coords'])
                            label = detector.get_class_name(d.get('cls', 0))
                            confidence = float(d.get('conf', d.get('confidence', 0.0)))
                            class_id = int(d.get('cls', 0))
                        else:
                            x1 = int(d.get('x1', 0)); y1 = int(d.get('y1', 0))
                            x2 = int(d.get('x2', 0)); y2 = int(d.get('y2', 0))
                            label = d.get('class', 'person')
                            confidence = float(d.get('confidence', 0.0))
                            class_id = 0
                        overlay_detections.append({
                            'box': [x1, y1, max(0, x2 - x1), max(0, y2 - y1)],
                            'label': label,
                            'confidence': confidence,
                            'class_id': class_id
                        })
                    if overlay_detections:
                        window_overlay.draw_detections_on_window(overlay_detections)
                except Exception:
                    pass

            # Per-track 5-frame face pipeline: process person detections
            # Also invoke minimal saver for ALL detections (not limited to persons)
            try:
                if stabilized_detections:
                    seen_tids = set()
                    for d in stabilized_detections:
                        bbox = detector._bbox_from_detection(d)
                        if not bbox:
                            continue
                        # Assign or create a tracking id for the bbox
                        tid, _ = detector._assign_or_create_track(bbox)
                        # Run face pipeline only for person class
                        if detector._is_person_detection(d):
                            seen_tids.add(tid)
                            detector._on_detection(tid, frame, bbox)
                        # Always run the minimal saver with per-track cooldown
                        try:
                            detector.on_detection(tid, frame, bbox)
                        except Exception as e:
                            print(f"{LOG_PREFIX} [on_detection ERROR] {e}", flush=True)
                    # Reset counters for tracks not seen (only affects person tracks)
                    try:
                        detector._init_foresight_tracking()
                        for tid in list(detector.fs_frames_seen.keys()):
                            if tid not in seen_tids:
                                detector.fs_frames_seen[tid] = 0
                    except Exception:
                        pass
            except Exception as e:
                print(f"{LOG_PREFIX}[PIPELINE]err:{e}", flush=True)
            
            # (Overlay already refreshed above to avoid any delay from burst pipeline)
            
            # Save frame if output specified
            if out:
                out.write(frame_with_detections)
            
            # Display if requested or in yolo-scrcpy-mode (but not if overlay mode)
            if (args.show or getattr(args, 'yolo_scrcpy_mode', False)) and not args.overlay:
                cv2.imshow('Foresight SAR Detection', frame_with_detections)
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
        # Clear global overlay reference
        GLOBAL_WINDOW_OVERLAY = None
        cv2.destroyAllWindows()
        print("[INFO] SAR detection terminated")

if __name__ == "__main__":
    main()