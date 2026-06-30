#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import time
from ultralytics import YOLO
from paddleocr import PaddleOCR

# ==========================================
# 1. YOLO Inference Class
# ==========================================
class YOLOProcessor:
    def __init__(self, model_path="book_seg.pt"):
        """Loads the YOLO model into memory."""
        print(f"[Info] Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)

    def detect_and_get_masks(self, image, min_area=200):
        """
        Runs YOLO inference on an image and returns valid bounding boxes and masks.
        """
        # Run prediction
        results = self.model.predict(
            source=image,
            classes=[0],
            retina_masks=True,
            half=True,
            device=0,
            verbose=False
        )

        if not results or results[0].masks is None:
            return []

        masks_data = results[0].masks.data.cpu().numpy()
        boxes_data = results[0].boxes.xyxy.cpu().numpy()
        
        detections = []
        for mask_float, box in zip(masks_data, boxes_data):
            mask_uint8 = (mask_float > 0.5).astype(np.uint8) * 255
            
            # Filter by area
            if np.count_nonzero(mask_uint8) < min_area:
                continue

            x1, y1, x2, y2 = map(int, box)
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "mask": mask_uint8
            })
            
        return detections


# ==========================================
# 2. PP-OCRv6 Inference Class
# ==========================================
class OCRProcessor:
    def __init__(self, device="gpu:0"):
        """Loads the PP-OCRv6 model into memory."""
        print("[Info] Loading PP-OCRv6 model...")
        self.ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv6_small_det",
            text_recognition_model_name="PP-OCRv6_small_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,  
            engine="transformers",
            device=device,
        )

    def extract_text(self, image):
        """
        Runs OCR on an image (numpy array or path) and returns a single concatenated string.
        """
        try:
            output = self.ocr.predict(input=image)
            full_text = []
            
            for res in output:
                # Based on the V6 json parsing structure from your script
                if hasattr(res, 'json') and "res" in res.json:
                    texts = res.json["res"].get("rec_texts", [])
                    full_text.extend(texts)
            
            return " ".join(full_text)
        except Exception as e:
            print(f"[OCR Error] Inference failed: {e}")
            return ""


# ==========================================
# 3. Combined Pipeline (YOLO + OCR)
# ==========================================
def process_books_pipeline(img_path, yolo_proc, ocr_proc, pad=2):
    """
    Combines YOLO segmentation and PP-OCRv6 to read text from isolated books.
    """
    t_start = time.time()
    
    # 1. Load Image
    bgr = cv2.imread(img_path)
    if bgr is None:
        print(f"[Error] Cannot find or read image: {img_path}")
        return []
    
    H, W = bgr.shape[:2]

    # 2. YOLO Inference
    detections = yolo_proc.detect_and_get_masks(bgr)
    print(f"[Result] Detected {len(detections)} books.")
    
    all_infos = []

    # 3. Process Each Detected Book
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        mask = det["mask"]

        # Apply padding
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(W, x2 + pad), min(H, y2 + pad)

        # Crop image and apply mask (set background to black)
        crop_img = bgr[y1:y2, x1:x2].copy()
        mask_crop = mask[y1:y2, x1:x2]
        crop_img[mask_crop == 0] = 0

        # Run OCR on the isolated, cropped book
        text = ocr_proc.extract_text(crop_img)

        all_infos.append({
            "bbox": (x1, y1, x2, y2),
            "text": text,
            "mask": mask
        })

    print(f"[Info] Pipeline completed in {time.time() - t_start:.2f} seconds.")
    return all_infos


# ==========================================
# Example Usage
# ==========================================
# if __name__ == "__main__":
#     # Initialize processors once (this prevents reloading models for every image)
#     yolo_processor = YOLOProcessor(model_path="book_seg.pt")
#     ocr_processor = OCRProcessor(device="gpu:0")

#     # Run the combined pipeline
#     results = process_books_pipeline(
#         img_path="book.png",
#         yolo_proc=yolo_processor,
#         ocr_proc=ocr_processor
#     )

#     # Print results
#     print("\n" + "=" * 30 + "\nFinal Results\n" + "=" * 30)
#     for i, info in enumerate(results):
#         print(f"\n--- Book #{i+1:02d} ---")
#         print(f"BBox: {info['bbox']}")
#         print(f"Text: {info['text']}")