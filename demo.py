from fastsam import FastSAM, FastSAMPrompt
import torch
import os
import cv2
import numpy as np
import random
from scipy.spatial import KDTree
from scipy.interpolate import splprep, splev
from PIL import Image
from skimage.feature import canny
from skimage.color import rgb2gray

def skimage_canny(image, sigma):
    img_gray = rgb2gray(image)
    # 高斯模糊  降低噪声
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    # Canny边缘检测
    edges = canny(img_gray, sigma)
    edge_points = np.column_stack(np.where(edges > 0))  # 得到边缘检测点
    edge_points = np.array(edge_points)
    return edge_points

def get_contours(ann):
    contour_all = []
    mask_all=[]
    for i in range(ann.shape[0]):
        mask = ann[i].cpu().numpy()
        mask = (mask * 255).astype(np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # if cv2.contourArea(contour) < 0.1 * mask.shape[0] * mask.shape[1]:  # 过滤太小的mask
            #     continue
            contour_all.append(contour)
        mask_all.append(mask)
    return contour_all,mask_all

# 插入点
def linear_interpolation(point1, point2, n):
    x1, y1 = point1
    x2, y2 = point2
    step_size = 1 / n
    interpolated_points = [(x1 + i * step_size * (x2 - x1), y1 + i * step_size * (y2 - y1)) for i in range(n + 1)]
    return interpolated_points

# 轮廓点中插入点
def interpolate_mask_edge_points(mask_edge_points):
    inter_mask_edge_points = []
    total_num_points = len(mask_edge_points)
    for i in range(total_num_points):
        current_point = np.array(mask_edge_points[i])
        next_point = np.array(mask_edge_points[(i + 1) % total_num_points])
        distance = np.linalg.norm(current_point - next_point)
        inter_points_num = max(int(distance / 4), 1)
        interpolated_points = linear_interpolation(current_point, next_point, inter_points_num)
        inter_mask_edge_points.extend(interpolated_points[:-1])
    inter_mask_edge_points = np.array(inter_mask_edge_points)
    return inter_mask_edge_points

# 去重
def unique_datas(points_data):
    unique_data, unique_indices = np.unique(points_data, axis=0, return_index=True)        # 去重
    unique_data_ordered = unique_data[np.argsort(unique_indices)]                          # 按原序排序
    return unique_data_ordered

def AdS(points, curvatures, threshold):
    points = points.T
    filtered_indices = curvatures > threshold
    filtered_points = points[:, filtered_indices]
    filtered_points = filtered_points.T
    return filtered_points

# B样条曲线拟合
def b_spline(points, k, node, num_fit):
    points = unique_datas(points)
    if len(points) <= k:
        return None, None, None, None
    try:
        x_values = points[:, 0]
        y_values = points[:, 1]
        if len(x_values) != len(y_values):
            raise ValueError("x_values 和 y_values 必须具有相同的长度")

        # knots = np.concatenate(([0] * k, np.linspace(0, 1, node - k + 1), [1] * k))

        try:
            if num_fit == 1:
            # 进行 B-spline 拟合
                tck, u = splprep([x_values, y_values], k=k, per=True)
                u_new = np.linspace(u.min(), u.max(), node)
                out = splev(u_new, tck)

                dx, dy = splev(u_new, tck, der=1)
                d2x, d2y = splev(u_new, tck, der=2)
                curvatures = np.abs(dx * d2y - dy * d2x) / (dx ** 2 + dy ** 2) ** 1.5
            elif num_fit == 2:
                tck, u = splprep([x_values, y_values], k=k, s=50, per=True)
                u_new = np.linspace(u.min(), u.max(), node)
                out = splev(u_new, tck)
                curvatures = None
            else:
                return None, None, None, None
        except TypeError as e:
            print(f"TypeError: {e}")
            return None, None, None, None
        except ValueError as e:
            print(f"ValueError: {e}")
            return None, None, None, None
        except Exception as e:
            print(f"OtherError: {e}")
            return None, None, None, None
    except Exception as e:
        print(f"Error occurred: {e}")
        return None, None, None, None

    # 曲线上的点
    fit_points = np.column_stack(out)
    return fit_points, out[0], out[1], curvatures


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

# 边缘点匹配
def match_dilated_boundary(boundary, ori_edge_points, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_boundary = cv2.dilate(boundary, kernel, iterations=1)
    # 获取 dilated_boundary 的高度和宽度
    height, width = dilated_boundary.shape
    edge_points = np.array(ori_edge_points)
    valid_mask = (edge_points[:, 0] >= 0) & (edge_points[:, 0] < height) & \
                 (edge_points[:, 1] >= 0) & (edge_points[:, 1] < width)
    valid_edge_points = edge_points[valid_mask]
    # 进一步筛选在膨胀边界内的点
    final_valid_mask = dilated_boundary[valid_edge_points[:, 0], valid_edge_points[:, 1]] > 0
    obj_edge_points = valid_edge_points[final_valid_mask]
    # 将满足条件的点添加到 obj_edge_points 中
    obj_edge_points = [(y, x) for x, y in obj_edge_points]            # 逆转坐标
    if len(obj_edge_points) == 0:
        return None

    obj_edge_points = np.array(obj_edge_points)
    return obj_edge_points

def get_thin_points(point_set, fit_points, sample_points, radius):
    points_num = len(fit_points)
    point_set = np.array(point_set)
    sample_points = set(map(tuple, sample_points))
    # 构建kd树
    kdtree_all = KDTree(point_set)
    kdtree_fit = KDTree(fit_points)
    thin_points = []
    processed_points = set()
    query_cache = {}

    for i, point in enumerate(point_set):
        if i > points_num:
            break
        # 将点转换为元组以便存储在集合中
        point_tuple = tuple(point)
        # 如果这个点已经在处理过的点集中，则跳过
        if point_tuple in processed_points:
            continue
        if point_tuple not in query_cache:
            query_cache[point_tuple] = kdtree_all.query_ball_point(point, radius)
        # 获取所有在范围内的点的索引
        points_indices_all = query_cache[point_tuple]
        # 获取范围内所有点的坐标
        points_in_radius = [point_set[idx] for idx in points_indices_all]

        if point_tuple in sample_points:
            if point_tuple not in query_cache:
                query_cache[point_tuple] = kdtree_fit.query_ball_point(point, radius)

            points_indices_fit = query_cache[point_tuple]
            # 获取范围内所有点的坐标
            points_in_radius_fit = [point_set[idx] for idx in points_indices_fit]
            set1 = set(map(tuple, points_in_radius))
            set2 = set(map(tuple, points_in_radius_fit))
            difference_set = set1 - set2
            if len(difference_set) > 0:
                # 将结果转换回数组形式
                result_points = np.array(list(difference_set))
                points_in_radius = [np.array(row, dtype='int64') for row in result_points]
        else:
            for idx in points_indices_all:
                processed_points.add(tuple(point_set[idx]))
        # 计算这些点的中心坐标
        if points_in_radius:
            center = tuple(map(lambda x: sum(x) / len(x), zip(*points_in_radius)))
            thin_points.append(center)
    return thin_points

def draw_mask(new_mask, mask_points):
    cv2.drawContours(new_mask, [mask_points], 0, (255, 255, 255), thickness=cv2.FILLED)  # 画出mask

    new_mask = Image.fromarray(new_mask, mode='L')
    mask_array = np.array(new_mask)
    return mask_array


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
