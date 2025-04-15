from ultralytics import YOLO
import cv2
import sys
# Load a model
if len(sys.argv) < 2:
    print('python segpredict.py pic_path')
    1/0
model = YOLO('/raid/platform/dwc/codes/yolo_sa/sa_test/train11/weights/last.pt')  # load a custom model
# model = YOLO('yolox_15_1024/last.pt')
# model = YOLO('yolom_15_e50.pt')
# Predict with the model
results = model(sys.argv[1], device='cpu', retina_masks=True, imgsz=1024, conf=0.01, iou=0.9, max_det=300, save_crop=True)  # predict on an image
# 为了画图改了ultralytics/yolo/engine/results.py 第221行
# import pdb;pdb.set_trace()
image = results[0].plot(labels=False, boxes=False, masks=True, probs=False)
# cv2.imwrite("yolox_15_1024_cake_pred_1024.jpg", image)
cv2.imwrite(sys.argv[1]+"_res.jpg", image)
