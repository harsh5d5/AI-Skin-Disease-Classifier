"""
Agent 2: Body Part Router
===========================
Uses a hybrid approach to detect which body part is in the image:
  1. OpenCV Haar Cascade (pre-trained) for Face detection
  2. Skin region analysis + aspect ratio heuristics for Torso vs Limbs

No GPU needed. All models are pre-trained and bundled with OpenCV.

Strategy:
  - Face detected prominently -> Route to Agent 3 (Facial)
  - Wide/square image with large skin area -> Route to Agent 4 (Torso)
  - Tall/narrow image or elongated skin region -> Route to Agent 5 (Limbs)
  - Default fallback -> Agent 4 (Torso - most common dermatology site)
"""

import os
import cv2
import numpy as np

# OpenCV comes with pre-trained Haar cascade models (no download needed!)
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
PROFILE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_profileface.xml'

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
profile_cascade = cv2.CascadeClassifier(PROFILE_CASCADE_PATH)


def _detect_face(image):
    """
    Detect faces using OpenCV's pre-trained Haar Cascade.
    Returns (detected: bool, confidence: float, face_area_ratio: float)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    img_area = h * w

    # Try frontal face detection
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        # Try profile face detection
        faces = profile_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

    if len(faces) > 0:
        # Get the largest face
        largest = max(faces, key=lambda f: f[2] * f[3])
        fx, fy, fw, fh = largest
        face_area = fw * fh
        face_ratio = face_area / img_area

        # Confidence based on face area ratio
        confidence = min(face_ratio * 5, 1.0)  # Scale up, cap at 1.0
        return True, confidence, face_ratio

    return False, 0.0, 0.0


def _detect_skin_region(image):
    """
    Detect skin-colored regions using HSV color space.
    Returns bounding box info and skin percentage.
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Skin color range in HSV (covers most skin tones)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Also check for darker skin tones
    lower_skin2 = np.array([0, 10, 50], dtype=np.uint8)
    upper_skin2 = np.array([25, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_skin, upper_skin)
    mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
    skin_mask = cv2.bitwise_or(mask1, mask2)

    # Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

    # Calculate skin percentage
    h, w = image.shape[:2]
    skin_pixels = cv2.countNonZero(skin_mask)
    skin_percentage = skin_pixels / (h * w)

    # Find contours of skin regions
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the largest skin region
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest_contour)
        aspect_ratio = cw / ch if ch > 0 else 1.0

        return {
            'skin_percentage': skin_percentage,
            'aspect_ratio': aspect_ratio,
            'region_width': cw / w,
            'region_height': ch / h,
            'region_y_center': (y + ch / 2) / h  # Vertical position
        }

    return {
        'skin_percentage': skin_percentage,
        'aspect_ratio': 1.0,
        'region_width': 0,
        'region_height': 0,
        'region_y_center': 0.5
    }


def _classify_body_part(skin_info, image_shape):
    """
    Classify body part based on skin region characteristics.
    """
    h, w = image_shape[:2]
    img_aspect = w / h
    region_aspect = skin_info['aspect_ratio']
    y_center = skin_info['region_y_center']

    # Scoring system
    torso_score = 0.0
    limbs_score = 0.0

    # Wide skin regions suggest torso (chest, back)
    if region_aspect > 1.2:
        torso_score += 0.3
    elif region_aspect < 0.6:
        limbs_score += 0.3  # Narrow/tall = arm or leg

    # Image aspect ratio
    if img_aspect > 1.1:  # Landscape
        torso_score += 0.2
    elif img_aspect < 0.8:  # Portrait/tall
        limbs_score += 0.2

    # Large skin area usually means torso
    if skin_info['skin_percentage'] > 0.5:
        torso_score += 0.2
    elif skin_info['skin_percentage'] < 0.3:
        limbs_score += 0.1

    # Vertical position: upper = more likely torso, lower = limbs
    if y_center < 0.4:
        torso_score += 0.1
    elif y_center > 0.6:
        limbs_score += 0.1

    # Normalize
    total = torso_score + limbs_score
    if total > 0:
        torso_score /= total
        limbs_score /= total
    else:
        torso_score = 0.5
        limbs_score = 0.5

    return torso_score, limbs_score


def route(img_path):
    """
    Main routing function.
    Analyzes the image and returns which specialist agent to use.

    Returns:
        dict with keys:
            - 'body_part': 'Face', 'Torso', or 'Limbs'
            - 'confidence': float (0-1)
            - 'route_to': 'agent3_facial', 'agent4_torso', or 'agent5_limbs'
            - 'reasoning': str explaining the decision
    """
    # Load image
    image = cv2.imread(str(img_path))
    if image is None:
        return {
            'body_part': 'Torso',
            'confidence': 0.0,
            'route_to': 'agent4_torso',
            'reasoning': 'Image could not be read. Defaulting to Torso specialist.'
        }

    # Step 1: Check for face
    face_detected, face_confidence, face_ratio = _detect_face(image)
    if face_detected and face_ratio > 0.08:  # Face occupies >8% of image
        return {
            'body_part': 'Face',
            'confidence': round(float(face_confidence), 3),
            'route_to': 'agent3_facial',
            'reasoning': f'Face detected (area: {face_ratio:.1%} of image). Routing to Facial Specialist.',
            'scores': {'Face': round(face_confidence, 3), 'Torso': 0.0, 'Limbs': 0.0}
        }

    # Step 2: Analyze skin region for Torso vs Limbs
    skin_info = _detect_skin_region(image)
    torso_score, limbs_score = _classify_body_part(skin_info, image.shape)

    if torso_score > limbs_score:
        return {
            'body_part': 'Torso',
            'confidence': round(float(torso_score), 3),
            'route_to': 'agent4_torso',
            'reasoning': f'Skin analysis: Torso={torso_score:.1%}, Limbs={limbs_score:.1%}. Wide skin region detected.',
            'scores': {'Face': 0.0, 'Torso': round(torso_score, 3), 'Limbs': round(limbs_score, 3)}
        }
    else:
        return {
            'body_part': 'Limbs',
            'confidence': round(float(limbs_score), 3),
            'route_to': 'agent5_limbs',
            'reasoning': f'Skin analysis: Torso={torso_score:.1%}, Limbs={limbs_score:.1%}. Elongated skin region detected.',
            'scores': {'Face': 0.0, 'Torso': round(torso_score, 3), 'Limbs': round(limbs_score, 3)}
        }


if __name__ == '__main__':
    import sys

    print("=" * 50)
    print("Agent 2: Body Part Router")
    print("Powered by OpenCV Haar Cascades (Pre-trained)")
    print("=" * 50)
    print("Routes: Face -> Agent 3 | Torso -> Agent 4 | Limbs -> Agent 5")
    print()

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        print(f"Analyzing: {img_path}")
        result = route(img_path)
        print(f"  Body Part: {result['body_part']}")
        print(f"  Confidence: {result['confidence']:.1%}")
        print(f"  Route To: {result['route_to']}")
        print(f"  Reasoning: {result['reasoning']}")
    else:
        print("Usage: python body_router.py <image_path>")
        print("Ready for integration with the pipeline.")
