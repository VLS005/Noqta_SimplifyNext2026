"""
Camera/vision integration.

Per the hackathon architecture decision, the camera is NEVER streamed
continuously - it only takes a triggered snapshot burst when this agent
calls one of these functions. Swap the bodies below for a real call to
Google Vision / a Bedrock vision model once the happy-path demo works.
"""

import cv2
import boto3
from typing import Optional
from domain.models import GPSPoint

def check_obstruction(position: GPSPoint) -> Optional[dict]:
    return None

def find_anchor_point(position: GPSPoint) -> Optional[dict]:
    # Capture frame using OpenCV
    cap = cv2.VideoCapture(1) # Try 1 first for Camo Studio
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        
    if not cap.isOpened():
        return {"object": "handrail (no camera)", "direction": "right", "distance_m": 2}
    
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return {"object": "handrail (camera read failed)", "direction": "right", "distance_m": 2}
    
    # AWS Rekognition expects bytes
    is_success, buffer = cv2.imencode(".jpg", frame)
    image_bytes = buffer.tobytes()
    
    # Call AWS Rekognition
    client = boto3.client('rekognition', region_name='us-east-1')
    target_label = "wall"
    
    try:
        response = client.detect_labels(Image={'Bytes': image_bytes}, MaxLabels=10, MinConfidence=70)
        labels = response.get('Labels', [])
        
        # Filter out generic labels
        for label in labels:
            name = label['Name']
            if name.lower() not in ["person", "human", "face", "clothing", "shoe", "people", "man", "woman"]:
                target_label = name.lower()
                
                # Draw bounding box on frame if it has instances
                if label.get('Instances'):
                    box = label['Instances'][0]['BoundingBox']
                    h, w, _ = frame.shape
                    left = int(box['Left'] * w)
                    top = int(box['Top'] * h)
                    width = int(box['Width'] * w)
                    height = int(box['Height'] * h)
                    cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 255), 4)
                    cv2.putText(frame, target_label, (left, max(top - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                break
                
        # Show the frame in a non-blocking popup
        cv2.imshow("Vision Agent (AWS Rekognition)", frame)
        cv2.waitKey(3000) # Show for 3 seconds
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Vision error: {e}")
        target_label = "landmark (vision error)"

    cap.release()
    return {"object": target_label, "direction": "right", "distance_m": 2}
