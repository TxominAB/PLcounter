import cv2
import numpy as np

def draw_bboxes(image, results):
    """Draw bounding boxes on the image and return the modified image along with the total count."""
    img_copy = image.copy()
    total_objects = 0

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            total_objects += 1
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img_copy, total_objects