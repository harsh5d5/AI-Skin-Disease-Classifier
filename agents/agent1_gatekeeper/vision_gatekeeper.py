"""
Agent 1: Vision Gatekeeper
============================
The first line of defense in the NeuralTrust pipeline.
Uses OpenCV to perform quality control on incoming skin images.

Responsibilities:
  1. Blur Detection      - Reject blurry/unfocused images (Laplacian variance)
  2. Watermark Detection - Reject images with logos/text overlays
  3. Brightness Check    - Reject overexposed or too-dark images
  4. Image Enhancement   - Sharpen and normalize accepted images
  5. Size Validation     - Ensure minimum resolution for CNN input

Flow:
  Input Image -> Gatekeeper -> [PASS/REJECT]
    PASS   -> Enhanced image sent to Agent 2 (Router)
    REJECT -> Error message returned with reason
"""

import os
import cv2
import numpy as np
from pathlib import Path


# ============================================================
# QUALITY THRESHOLDS
# ============================================================
MIN_SHARPNESS = 50.0       # Laplacian variance threshold (lower = more blurry)
MIN_BRIGHTNESS = 30        # Minimum average pixel brightness (0-255)
MAX_BRIGHTNESS = 240       # Maximum average pixel brightness (0-255)
MIN_RESOLUTION = 64        # Minimum width/height in pixels
LOGO_AREA_THRESHOLD = 0.01 # Max white blob area ratio (1% of image)
TEXT_EDGE_THRESHOLD = 1.5   # Edge density threshold for text detection


class VisionGatekeeper:
    """
    OpenCV-powered image quality controller.
    Ensures only clean, high-quality images reach the specialist agents.
    """

    def __init__(self, config=None):
        """Initialize with optional custom thresholds."""
        self.min_sharpness = MIN_SHARPNESS
        self.min_brightness = MIN_BRIGHTNESS
        self.max_brightness = MAX_BRIGHTNESS
        self.min_resolution = MIN_RESOLUTION

        if config:
            self.min_sharpness = config.get('min_sharpness', MIN_SHARPNESS)
            self.min_brightness = config.get('min_brightness', MIN_BRIGHTNESS)
            self.max_brightness = config.get('max_brightness', MAX_BRIGHTNESS)
            self.min_resolution = config.get('min_resolution', MIN_RESOLUTION)

    # --------------------------------------------------------
    # QUALITY CHECKS
    # --------------------------------------------------------

    def check_sharpness(self, image):
        """
        Measure image sharpness using Laplacian variance.
        Higher value = sharper image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        score = laplacian.var()
        return {
            'score': round(float(score), 2),
            'passed': bool(score >= self.min_sharpness),
            'reason': f'Sharpness: {score:.1f} (min: {self.min_sharpness})'
        }

    def check_brightness(self, image):
        """
        Check if image brightness is within acceptable range.
        Rejects overexposed (washed out) and underexposed (too dark) images.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        passed = self.min_brightness <= mean_brightness <= self.max_brightness
        return {
            'score': round(float(mean_brightness), 2),
            'passed': bool(passed),
            'reason': f'Brightness: {mean_brightness:.0f} (range: {self.min_brightness}-{self.max_brightness})'
        }

    def check_resolution(self, image):
        """
        Ensure image meets minimum resolution requirements.
        """
        h, w = image.shape[:2]
        passed = h >= self.min_resolution and w >= self.min_resolution
        return {
            'width': int(w),
            'height': int(h),
            'passed': bool(passed),
            'reason': f'Resolution: {w}x{h} (min: {self.min_resolution}x{self.min_resolution})'
        }

    def check_watermark(self, image):
        """
        Detect large white logos/watermarks using connected component analysis.
        Also checks for text-heavy corners (common in medical image databases).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 1. Check for large white blobs (DermNet-style logos)
        _, white_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white_mask)

        img_area = h * w
        has_logo = False

        for i in range(1, num_labels):
            blob_area = stats[i, cv2.CC_STAT_AREA]
            if blob_area > img_area * LOGO_AREA_THRESHOLD:
                has_logo = True
                break

        # 2. Check corners for text watermarks (edge density analysis)
        edges = cv2.Canny(gray, 100, 200)
        corner_h = int(h * 0.15)
        top_edges = np.sum(edges[0:corner_h, :])
        bottom_edges = np.sum(edges[h - corner_h:, :])
        corner_edges = top_edges + bottom_edges
        has_text = corner_edges > (img_area * TEXT_EDGE_THRESHOLD)

        passed = not has_logo and not has_text

        if has_logo:
            reason = 'Large white logo/watermark detected'
        elif has_text:
            reason = 'Text watermark detected in image corners'
        else:
            reason = 'No watermarks detected'

        return {
            'has_logo': bool(has_logo),
            'has_text': bool(has_text),
            'passed': bool(passed),
            'reason': reason
        }

    # --------------------------------------------------------
    # IMAGE ENHANCEMENT
    # --------------------------------------------------------

    def enhance_image(self, image):
        """
        Apply gentle enhancement to improve image quality for CNN processing.
        - Light sharpening (Unsharp Mask)
        - CLAHE for contrast normalization
        - Noise reduction
        """
        # Step 1: Light noise reduction
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)

        # Step 2: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # Step 3: Gentle sharpening kernel
        sharpen_kernel = np.array([
            [0,  -0.5, 0],
            [-0.5, 3, -0.5],
            [0,  -0.5, 0]
        ], dtype=np.float32)
        sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)

        return sharpened

    # --------------------------------------------------------
    # MAIN GATE FUNCTION
    # --------------------------------------------------------

    def process(self, img_path, enhance=True):
        """
        Main gatekeeper function. Runs all quality checks on an image.

        Args:
            img_path: Path to the input image
            enhance: Whether to apply enhancement on passed images

        Returns:
            dict with keys:
                - 'status': 'PASS' or 'REJECT'
                - 'image': Enhanced image (numpy array) if PASS, None if REJECT
                - 'original_path': Original file path
                - 'checks': Dict of all quality check results
                - 'rejection_reason': Reason for rejection (if REJECT)
                - 'quality_score': Overall quality score (0-100)
        """
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            return {
                'status': 'REJECT',
                'image': None,
                'original_path': str(img_path),
                'checks': {},
                'rejection_reason': 'Image could not be loaded (corrupted or unsupported format)',
                'quality_score': 0
            }

        # Run all quality checks
        sharpness = self.check_sharpness(image)
        brightness = self.check_brightness(image)
        resolution = self.check_resolution(image)
        watermark = self.check_watermark(image)

        checks = {
            'sharpness': sharpness,
            'brightness': brightness,
            'resolution': resolution,
            'watermark': watermark
        }

        # Calculate overall quality score (0-100)
        scores = []
        if sharpness['passed']:
            scores.append(min(sharpness['score'] / 200 * 100, 100))
        else:
            scores.append(sharpness['score'] / 200 * 50)

        if brightness['passed']:
            scores.append(80)
        else:
            scores.append(20)

        if resolution['passed']:
            scores.append(90)
        else:
            scores.append(10)

        if watermark['passed']:
            scores.append(100)
        else:
            scores.append(0)

        quality_score = round(float(np.mean(scores)), 1)

        # Determine PASS or REJECT
        all_passed = all([
            sharpness['passed'],
            brightness['passed'],
            resolution['passed'],
            watermark['passed']
        ])

        if all_passed:
            # Apply enhancement if requested
            output_image = self.enhance_image(image) if enhance else image

            return {
                'status': 'PASS',
                'image': output_image,
                'original_path': str(img_path),
                'checks': checks,
                'rejection_reason': None,
                'quality_score': quality_score
            }
        else:
            # Find the rejection reason
            reasons = []
            if not sharpness['passed']:
                reasons.append(f"Blurry image ({sharpness['score']})")
            if not brightness['passed']:
                reasons.append(f"Bad brightness ({brightness['score']})")
            if not resolution['passed']:
                reasons.append(f"Too small ({resolution.get('width', '?')}x{resolution.get('height', '?')})")
            if not watermark['passed']:
                reasons.append(watermark['reason'])

            return {
                'status': 'REJECT',
                'image': None,
                'original_path': str(img_path),
                'checks': checks,
                'rejection_reason': ' | '.join(reasons),
                'quality_score': quality_score
            }


# Convenience function for pipeline integration
def process(img_path, enhance=True):
    """Quick-access function for the pipeline."""
    gatekeeper = VisionGatekeeper()
    return gatekeeper.process(img_path, enhance)


if __name__ == '__main__':
    import sys

    print("=" * 55)
    print("  Agent 1: Vision Gatekeeper")
    print("  Powered by OpenCV (Pre-trained Algorithms)")
    print("=" * 55)
    print()
    print("Quality Checks:")
    print(f"  Sharpness   : Laplacian variance >= {MIN_SHARPNESS}")
    print(f"  Brightness  : {MIN_BRIGHTNESS} - {MAX_BRIGHTNESS}")
    print(f"  Resolution  : >= {MIN_RESOLUTION}x{MIN_RESOLUTION}px")
    print(f"  Watermark   : Logo area < {LOGO_AREA_THRESHOLD*100}% of image")
    print()

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        print(f"Analyzing: {img_path}")
        result = process(img_path)
        print(f"  Status: {result['status']}")
        print(f"  Quality Score: {result['quality_score']}/100")
        if result['rejection_reason']:
            print(f"  Rejection: {result['rejection_reason']}")
        for name, check in result['checks'].items():
            status = 'PASS' if check['passed'] else 'FAIL'
            print(f"  [{status}] {check['reason']}")
    else:
        print("Usage: python vision_gatekeeper.py <image_path>")
        print("Ready for pipeline integration.")
