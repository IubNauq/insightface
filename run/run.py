import cv2
import sys
import numpy as np

sys.path.append("../insightface/src_recognition")
import main
from scrfd import SCRFD
from arcface_onnx import ArcFaceONNX

# MODEL PATH
DET_MODEL_PATH = "./detection_model/det_2.5g.onnx"
REC_MODEL_PATH = "./recognition_model/glintasia_r50.onnx"

# INITIALIZE detect model
det = SCRFD(DET_MODEL_PATH)
det.prepare(0)

# INITIALIZE recognition model
rec = ArcFaceONNX(REC_MODEL_PATH)
rec.prepare(0)


def is_blurry(image, box, threshold=100):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    box = np.round(box).astype(np.int32)
    x1, y1, x2, y2, _ = box
    roi = gray[y1:y2, x1:x2]

    # Compute the Laplacian of the image
    laplacian_var = cv2.Laplacian(roi, cv2.CV_64F).var()
    print("LAP: ", laplacian_var)
    # Compare the variance with the threshold
    return laplacian_var < threshold


if __name__ == "__main__":
    # IMAGE PATH
    img1_path = "..."
    img2_path = "..."

    # CALCULATE cosine similarity
    sim, conclu = main.func(
        img1_path,
        img2_path,
        det,
        rec,
    )

    # CALCULATE bounding box, kps, feature vector
    box, kps, feat = main.get_feature(
        img1_path,
        det,
        rec,
    )
