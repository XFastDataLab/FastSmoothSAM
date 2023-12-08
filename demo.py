from fastsam import FastSAM, FastSAMPrompt
import torch
import os
from fastsmoothsam import *

s = 100            # B样条的平滑程度
k = 3              # B样条阶数
node = 200         # B样条的节点数

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FastSAM('../weights/FastSAM.pt')
imgpath = './images/'
outpath = './output/'
files = os.listdir(imgpath)

everything_results = model(device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)

for file in files:
    img_path = imgpath + file
    img = cv2.imread(img_path)
    try:
        everything_result = everything_results(img_path, stream=False)
        prompt_process = FastSAMPrompt(img, img_path, everything_result, device=DEVICE)
        ann = prompt_process.everything_prompt()
        prompt_process.plot(img, annotations=ann, output=outpath, )  # fastSAM得到的彩色图


    except Exception as e:
        print(e)
        continue

