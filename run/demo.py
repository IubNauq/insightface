import cv2
import numpy as np
import os
import sys
import pickle
import time
import math

sys.path.append("../insightface/src_recognition")

import main
from arcface_onnx import ArcFaceONNX
from scrfd import SCRFD

# sys.path.append("/home/buiquan/Desktop/insightface/insightface/python-package/insightface")
from insightface.app import FaceAnalysis
from scipy.spatial import distance as dist


# MODEL PATH
DET_MODEL_PATH = "./detection_model/det_2.5g.onnx"
REC_MODEL_PATH = "./recognition_model/glintasia_r50.onnx"

SIM_THRESHOLD = 0.28
EYE_THRESHOLD = 0.15

FACE_RL_THRESHOLD = 2
FACE_UP_THRESHOLD = 0.25
FACE_DOWN_THRESHOLD = 0.55
TRUE_BOX_COLOR = (0, 255, 0)
WRONG_BOX_COLOR = (0, 0, 255)
BOX_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 1
TEXT_THICKNESS = 2
LANDMARK_COLOR = (200, 160, 75)
LANDMARK_RADIUS = 1
LANDMARK_THICKNESS = 1

# LOAD detect model
det = SCRFD(DET_MODEL_PATH)
det.prepare(0)

# LOAD recognition model
rec = ArcFaceONNX(REC_MODEL_PATH)
rec.prepare(0)

# LOAD app for landmark
app = FaceAnalysis(allowed_modules=["detection", "landmark_2d_106"])
app.prepare(ctx_id=0, det_size=(320, 320), det_thresh=0.6)


def image_infer(img):
    if isinstance(img, str):
        img = cv2.imread(img)

    # start = time.time()
    faces = app.get(img, max_num=1)
    # assert len(faces)==6
    # tim = img.copy()
    box = []
    kps = []
    lmk = []
    for face in faces:
        lmk = face.landmark_2d_106
        box = face.bbox
        kps = face.kps
        print("det_score: ", face.det_score)
    # print("Landmark time: ", time.time() - start)
    return box, kps, lmk


def face_analysis(lmk):
    # EYE Analysis
    eye41, eye42, eye36, eye37 = lmk[41], lmk[42], lmk[36], lmk[37]
    eye35, eye39 = lmk[35], lmk[39]
    A = dist.euclidean(eye41, eye36)
    B = dist.euclidean(eye42, eye37)
    C = dist.euclidean(eye35, eye39)
    lefteye_rate = (A + B) / (2.0 * C)

    eye95, eye90, eye96, eye91 = lmk[95], lmk[90], lmk[96], lmk[91]
    eye89, eye93 = lmk[89], lmk[93]
    A = dist.euclidean(eye95, eye90)
    B = dist.euclidean(eye96, eye91)
    C = dist.euclidean(eye89, eye93)
    righteye_rate = (A + B) / (2.0 * C)

    eye_rate = (lefteye_rate + righteye_rate) / 2.0
    result = ""
    if eye_rate <= EYE_THRESHOLD:
        result += f"EYE CLOSE {eye_rate}"
    else:
        result += f"EYE OPEN {eye_rate}"

    # FACE Analysis
    face12, face28 = lmk[12], lmk[28]
    nose86 = lmk[86]
    A = dist.euclidean(face12, nose86)
    B = dist.euclidean(face28, nose86)
    face_RL_rate = A / B
    if face_RL_rate >= FACE_RL_THRESHOLD:
        result += f" FACE RIGHT {face_RL_rate}"
    elif 1 / face_RL_rate >= FACE_RL_THRESHOLD:
        result += f" FACE LEFT {face_RL_rate}"

    chin0 = lmk[0]
    forehead72 = lmk[72]
    A = dist.euclidean(forehead72, nose86)
    B = dist.euclidean(chin0, nose86)

    face_UD_rate = A / B
    if face_UD_rate <= FACE_UP_THRESHOLD:
        result += f" FACE UP {face_UD_rate}"
    elif face_UD_rate >= FACE_DOWN_THRESHOLD:
        result += f" FACE DOWN {face_UD_rate}"
    return result


if __name__ == "__main__":
    cam = cv2.VideoCapture(0)

    # GET vector data from pickle file
    with open(
        "./vector_data/glintasia_d2.5g.pkl",
        "rb",
    ) as handle:
        vector_data = pickle.load(handle)

    # GET label data from pickle file
    with open("./label_data/label.pkl", "rb") as handle:
        label = pickle.load(handle)

    while True:
        ret, img = cam.read()

        box, kps, lmk = image_infer(img)
        feat = []
        if len(kps):
            feat = rec.get(img, kps)
        if len(lmk):
            print(face_analysis(lmk))

        if len(feat):
            # SEARCH similar vector
            highest_sim = 0
            ID = -1
            for i in range(len(vector_data)):
                sim = rec.compute_sim(feat, vector_data[i])
                if sim > highest_sim:
                    highest_sim = sim
                    ID = i
            if highest_sim >= SIM_THRESHOLD:
                predict_label = label[ID]
                box_color = TRUE_BOX_COLOR
            else:
                predict_label = "Unknown"
                box_color = WRONG_BOX_COLOR

            # DRAW Box and Text
            if len(box):
                box = np.round(box).astype(np.int64)
                for x1, y1, x2, y2 in [box]:
                    cv2.rectangle(
                        img,
                        (x1, y1),
                        (x2, y2),
                        box_color,
                        BOX_THICKNESS,
                    )

                    cv2.putText(
                        img,
                        f"{str(predict_label)} {str(round((highest_sim+1)/2, 2))}",
                        (x1, y2),
                        TEXT_FONT,
                        TEXT_SCALE,
                        TEXT_COLOR,
                        TEXT_THICKNESS,
                    )

            # DRAW Landmarks
            lmk = np.round(lmk).astype(np.int64)
            for i in range(lmk.shape[0]):
                p = tuple(lmk[i])
                cv2.circle(
                    img,
                    p,
                    LANDMARK_RADIUS,
                    LANDMARK_COLOR,
                    LANDMARK_THICKNESS,
                    cv2.LINE_AA,
                )

        cv2.imshow("img", img)
        # If 'q' is pressed, close program
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    # Stop the camera
    cam.release()
    # Close all windows
    cv2.destroyAllWindows()
