import cv2
import numpy as np
import threading
import time
import requests
from tensorflow import keras
from django.conf import settings
import logging
from collections import deque

logger = logging.getLogger(__name__)

class PhoneDetector:
    def __init__(self):
        self.model = None
        self.is_running = False
        self.detection_thread = None
        self.camera = None
        self.user_id = None
        self.last_detection_time = 0
        self.detection_cooldown = 2.0 
        self.last_confidence_phone = 0
        self.last_confidence_no_phone = 0
        self.frame_count = 0
        self.start_time = time.time()
        

        self.confidence_threshold = 0.5
        self.required_detections = 2
        self.required_duration = 2
        

        self.confidence_history = deque(maxlen=10)
        self.high_confidence_count = 0
        self.high_confidence_start_time = None
        self.detection_buffer = deque(maxlen=self.required_detections)
            
    def load_model(self, model_path):
        try:
            print(model_path)
            if model_path:
                self.model = keras.models.load_model(model_path)
            else:
                self.model = keras.models.load_model('keras_model.h5')

            print(f"!!😊😊🥀🥀😊😊Phone detection model loaded successfully at {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def preprocess_frame(self, frame):

        resized = cv2.resize(frame, (224, 224))

        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        normalized = rgb_frame.astype(np.float32) / 255.0

        batch = np.expand_dims(normalized, axis=0)
        return batch
    
    def should_trigger_detection(self):

        current_time = time.time()
        

        high_conf_count = sum(1 for conf in self.detection_buffer if conf >= self.confidence_threshold)
        

        duration_met = False
        if self.high_confidence_start_time and current_time - self.high_confidence_start_time >= self.required_duration:
            duration_met = True
        

        avg_confidence = np.mean(list(self.detection_buffer)) if self.detection_buffer else 0
        
        debug_info = {
            'high_conf_count': high_conf_count,
            'required_detections': self.required_detections,
            'duration_sustained': current_time - self.high_confidence_start_time if self.high_confidence_start_time else 0,
            'required_duration': self.required_duration,
            'avg_confidence': avg_confidence,
            'buffer_size': len(self.detection_buffer),
            'buffer_contents': list(self.detection_buffer)
        }
        

        should_trigger = (high_conf_count >= self.required_detections and duration_met)
        
        return should_trigger, debug_info
    
    def detect_phone_in_frame(self, frame):

        if self.model is None:
            print("❌ MODEL IS NONE!")
            return False, {}
            
        try:
            processed_frame = self.preprocess_frame(frame)

            

            predictions = self.model.predict(processed_frame, verbose=0)


            self.last_confidence_phone = float(predictions[0][0])
            self.last_confidence_no_phone = float(predictions[0][1])
            self.frame_count += 1
            

            self.confidence_history.append(self.last_confidence_phone)
            self.detection_buffer.append(self.last_confidence_phone)
            
            current_time = time.time()


            if self.last_confidence_phone >= self.confidence_threshold:
                if self.high_confidence_start_time is None:
                    self.high_confidence_start_time = current_time
                self.high_confidence_count += 1

            else:

                if self.high_confidence_start_time is not None:
                    print("🔄 Confidence dropped - resetting sustained detection tracking")
                self.high_confidence_start_time = None
                self.high_confidence_count = 0
            

            should_trigger, debug_info = self.should_trigger_detection()

            
            return should_trigger, debug_info
            
        except Exception as e:
            print(f"❌ DETECTION EXCEPTION: {e}")
            logger.error(f"Error during phone detection: {e}")
            return False, {}
    
    def reset_detection_state(self):

        print("🔄 Resetting detection state after successful trigger")
        self.high_confidence_start_time = None
        self.high_confidence_count = 0
        self.detection_buffer.clear()

    
    def notify_django_backend(self, debug_info=None):

        current_time = time.time()
        
        # Implement cooldown to prevent spam notifications
        if current_time - self.last_detection_time < self.detection_cooldown:
            print(f"⏰ Detection cooldown active ({current_time - self.last_detection_time:.1f}s < {self.detection_cooldown}s)")
            return
            
        self.last_detection_time = current_time
        
        try:

            payload = {
                'user_id': self.user_id,
                'confidence': self.last_confidence_phone,
                'debug_info': debug_info or {}
            }
            

            response = requests.post(
                'http://127.0.0.1:8000/api/phone_detected/',  
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=2
            )
            if response.status_code == 200:
                logger.info("Phone detection reported to Django backend")
                print("✅ Successfully notified Django backend")

                self.reset_detection_state()
            else:
                logger.warning(f"Failed to report detection: {response.status_code}")
                print(f"⚠️ Backend response: {response.status_code}")
        except requests.RequestException as e:
            logger.error(f"Failed to notify backend: {e}")
            print(f"❌ Backend notification failed: {e}")
    
    def get_detection_stats(self):

        current_time = time.time()
        
        return {
            'confidence_threshold': self.confidence_threshold,
            'required_detections': self.required_detections,
            'required_duration': self.required_duration,
            'current_confidence': self.last_confidence_phone,
            'confidence_history': list(self.confidence_history),
            'high_confidence_count': self.high_confidence_count,
            'sustained_duration': current_time - self.high_confidence_start_time if self.high_confidence_start_time else 0,
            'detection_buffer': list(self.detection_buffer),
            'avg_recent_confidence': np.mean(list(self.detection_buffer)) if self.detection_buffer else 0,
            'is_tracking_sustained': self.high_confidence_start_time is not None
        }
    
    def detection_loop(self):

        logger.info("Starting enhanced phone detection loop")
        
        # Initialize camera
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                print("❌ CAMERA FAILED TO OPEN")
                logger.error("Failed to open camera")
                return
                
            ret, test_frame = self.camera.read()
            if not ret or test_frame is None:
                print("❌ CAMERA READ FAILED")
                logger.error("Failed to read from camera")
                self.camera.release()
                return
            logger.info("Camera opened successfully")
            
        except Exception as e:
            print(f"❌ CAMERA EXCEPTION: {e}")
            logger.error(f"Exception while opening camera: {e}")
            return
        

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 10)
        
        frame_skip = 3
        frame_count = 0
        consecutive_failures = 0

        while self.is_running:
            ret, frame = self.camera.read()
            if not ret or frame is None:
                consecutive_failures += 1
                print(f"❌ Frame capture failed (attempt {consecutive_failures})")
                if consecutive_failures > 10:
                    print("❌ TOO MANY FAILURES - BREAKING")
                    break
                time.sleep(0.1)
                continue
            
            consecutive_failures = 0
            frame_count += 1
            

            if frame_count % frame_skip != 0:
                continue

            

            try:
                should_trigger, debug_info = self.detect_phone_in_frame(frame)

                
                if should_trigger:

                    logger.info("Sustained phone detection confirmed!")
                    self.notify_django_backend(debug_info)
                else:

                    stats = self.get_detection_stats()

            except Exception as e:
                print(f"❌ DETECTION ERROR: {e}")
                logger.error(f"Error in detection loop: {e}")
            
            time.sleep(0.1)

        if self.camera:
            self.camera.release()
        logger.info("Enhanced phone detection loop stopped")
    
    def start_detection(self, user_id):

        if self.is_running:
            logger.warning("Detection already running")
            return False
        
        if self.model is None:
            logger.error("Model not loaded - cannot start detection")
            return False
        
        self.user_id = user_id
        self.is_running = True
        

        self.reset_detection_state()
        self.confidence_history.clear()
        

        self.detection_thread = threading.Thread(target=self.detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        logger.info(f"Enhanced phone detection started for user {user_id}")
        return True
    
    def stop_detection(self):

        if not self.is_running:
            return
        
        logger.info("Stopping enhanced phone detection...")
        self.is_running = False
        

        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=3.0)
            if self.detection_thread.is_alive():
                logger.warning("Detection thread did not stop gracefully")
        

        self.reset_detection_state()
        self.confidence_history.clear()
        self.user_id = None
        logger.info("Enhanced phone detection stopped")

# Global instance
phone_detector = PhoneDetector()