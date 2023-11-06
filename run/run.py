import cv2
import sys

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

if __name__ == "__main__":
    # IMAGE PATH
    img1_path = "..."
    img2_path = "..."

    # CALCULATE cosine similarity, 
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

