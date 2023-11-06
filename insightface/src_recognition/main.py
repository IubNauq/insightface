import cv2
import onnxruntime
# from scrfd import SCRFD
# from arcface_onnx import ArcFaceONNX

onnxruntime.set_default_logger_severity(3)

# assets_dir = osp.expanduser('~/.insightface/models/buffalo_l')
# detector = SCRFD(os.path.join(assets_dir, 'det_10g.onnx'))
# model_path = os.path.join(assets_dir, 'w600k_r50.onnx')

# Compare two image are the same person or not!
def func(img1, img2, detector, rec):
    if isinstance(img1, str):
        image1 = cv2.imread(img1)
    else:
        image1 = img1

    if isinstance(img2, str):
        image2 = cv2.imread(img2)
    else:
        image2 = img2


    bboxes1, kpss1 = detector.autodetect(image1, max_num=1)
    if bboxes1.shape[0] == 0:
        return -1.0, "Face not found in Image-1"
    bboxes2, kpss2 = detector.autodetect(image2, max_num=1)
    if bboxes2.shape[0] == 0:
        return -1.0, "Face not found in Image-2"

    kps1 = kpss1[0]
    kps2 = kpss2[0]

    feat1 = rec.get(image1, kps1)
    feat2 = rec.get(image2, kps2)
    sim = rec.compute_sim(feat1, feat2)
    if sim < 0.2:
        conclu = "They are NOT the same person"

    elif sim >= 0.2 and sim < 0.28:
        conclu = "They are LIKELY TO be the same person"
    else:
        conclu = "They ARE the same person"
    return sim, conclu


# Get box, kps, feature of an image
def get_feature(img, det, rec):
    if isinstance(img, str):
        img = cv2.imread(img)
    bboxes, kpss = det.autodetect(img, max_num=1)
    if bboxes.shape[0] == 0:
        print(f"Face not found in Image")
        return [], [], []

    kps = kpss[0]
    feat = rec.get(img, kps)
    return bboxes[0], kpss[0], feat
