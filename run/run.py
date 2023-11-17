import cv2
import sys
import numpy as np

sys.path.append("../insightface/src_recognition")
import main
from scrfd import SCRFD
from arcface_onnx import ArcFaceONNX

sys.path.append("../insightface/insightface")
from app.face_analysis import FaceAnalysis


# MODEL PATH
DET_MODEL_PATH = "./detection_model/det_2.5g.onnx"
REC_MODEL_PATH = "./recognition_model/glintasia_r50.onnx"

# INITIALIZE detect model
det = SCRFD(DET_MODEL_PATH)
det.prepare(0)

# INITIALIZE recognition model
rec = ArcFaceONNX(REC_MODEL_PATH)
rec.prepare(0)

app = FaceAnalysis(allowed_modules=["detection", "genderage"])
app.prepare(ctx_id=0, det_size=(320, 320))


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


def estimate_genderage(img_path):
    img = cv2.imread(img_path)
    res = app.get(img)
    gender, age = res[0].sex, res[0].age
    return gender, age


if __name__ == "__main__":
    # Estimate gender, age
    img_path = "..."
    gender, age = estimate_genderage(img_path)
    
    
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
