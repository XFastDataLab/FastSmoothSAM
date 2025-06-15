from fastsam import FastSAM, FastSAMPrompt
import torch
import os
import cv2
import numpy as np
import random
from PIL import Image

# 将所有mask画在一张图上
def make_color_masks(image, masks_array):
    composite_image = Image.new("RGBA", image.size)
    for mask_array in masks_array:
        random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 128)
        mask_color_image = Image.new("RGBA", image.size, random_color)
        mask_img = Image.fromarray(mask_array, mode='L')
        composite_image.paste(mask_color_image, (0, 0), mask_img)
    final_image = Image.alpha_composite(image, composite_image)
    result_array = np.array(final_image)
    for mask_array in masks_array:
        _, binary_mask = cv2.threshold(mask_array, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result_array, contours, -1, (0, 0, 255, 255), 2)
    result_array = cv2.cvtColor(result_array, cv2.COLOR_BGR2RGB)
    return result_array

def main(img,imgpath,outpath_dir,file):
    # try:
        everything_result = everything_results(imgpath, stream=False)
        prompt_process = FastSAMPrompt(img, imgpath, everything_result, device=DEVICE)
        ann = prompt_process.everything_prompt()

        masks_array =ann
        image_PLI = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        image_PLI = Image.fromarray(image_PLI)
        result_image = make_color_masks(image_PLI, masks_array)          # 画出叠加在一起的mask彩色图

        cv2.imwrite(outpath_dir + file, result_image)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FastSAM('./weights/FastSmoothSAM.pt')
    everything_results = model(device=DEVICE, retina_masks=True, imgsz=1024, conf=0.5, iou=0.9, )

    imgpath_dir = 'images/'
    outpath_dir = 'output/'
    files = os.listdir(imgpath_dir)
    for file in files:
        imgpath = imgpath_dir + file
        img = cv2.imread(imgpath)
        main(img,imgpath,outpath_dir,file)
