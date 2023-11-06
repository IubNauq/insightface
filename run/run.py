import cv2
import sys

sys.path.append("/home/buiquan/Desktop/insightface/insightface/src_recognition/")
import main
from scrfd import SCRFD
from arcface_onnx import ArcFaceONNX

# MODEL PATH
DET_MODEL_PATH = "./detection_model/det_2.5g.onnx"
REC_MODEL_PATH = "./recognition_model/glintasia_r50.onnx"

# LOAD detect model
det = SCRFD(DET_MODEL_PATH)
det.prepare(0)

# LOAD recognition model
rec = ArcFaceONNX(REC_MODEL_PATH)
rec.prepare(0)

if __name__ == "__main__":
    # IMAGE PATH
    img1_path = "/home/buiquan/Downloads/TelegramDesktop/recognition/P2300378/1696386098.134238.png"
    img2_path = "/home/buiquan/Downloads/TelegramDesktop/recognition/P2300054/1694068026.922021.png"

    # GET cosine similarity, 
    sim, conclu = main.func(
        img1_path,
        img2_path,
        det,
        rec,
    )

    # GET bounding box, kps, feature vector
    box, kps, feat = main.get_feature(
        img1_path,
        det,
        rec,
    )

    print("SIM: ", sim)
    print("Conclu: ", conclu)
    print("BOX: ", box)
    print("KPS: ", kps)
    print("Num dimension of Feature vector: ", len(feat))
