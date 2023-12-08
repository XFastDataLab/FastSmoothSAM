import cv2
import numpy as np
import random
from scipy.spatial import KDTree
from scipy.interpolate import splprep, splev
from PIL import Image



def get_canny_points(img_gray):
    edges = cv2.Canny(image=img_gray, threshold1=100, threshold2=200)  # Canny边缘检测
    edge_points = np.column_stack(np.where(edges > 0))  # 得到边缘检测点
    return edge_points

def get_contours(ann):
    contour_all = []
    for i in range(ann.shape[0]):
        mask = ann[i].cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour_all.append(contour)
    return contour_all

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
        inter_points_num = max(int(distance / 2), 1)
        interpolated_points = linear_interpolation(current_point, next_point, inter_points_num)
        inter_mask_edge_points.extend(interpolated_points[:-1])
    inter_mask_edge_points = np.array(inter_mask_edge_points)
    return inter_mask_edge_points
# 去重
def unique_datas(points_data):
    unique_data, unique_indices = np.unique(points_data, axis=0, return_index=True)        # 去重
    unique_data_ordered = unique_data[np.argsort(unique_indices)]                          # 按原序排序
    return unique_data_ordered

# B样条曲线拟合
def b_spline(points, k, s, node):
    points = unique_datas(points)
    if len(points) <= k:
        return
    x_values = np.array([point[0] for point in points])
    y_values = np.array([point[1] for point in points])

    # 进行 B-spline 拟合
    tck, u = splprep([x_values, y_values], k=k, s=s, per=True)
    u_new = np.linspace(u.min(), u.max(), node)
    x_fit, y_fit = splev(u_new, tck)
    fit_points = np.column_stack((x_fit, y_fit)).astype(np.int32)
    fit_points = unique_datas(fit_points)
    return fit_points

def find_center_points(points_A, points_B, radius):
    # 构建kd树
    kdtree = KDTree(points_B)
    center_points = []
    for point_a in points_A:
        # 获取所有在范围内的点的索引
        points_indices = kdtree.query_ball_point(point_a, radius)
        # 获取范围内所有点的坐标
        points_in_radius = [points_B[idx] for idx in points_indices]
        points_in_radius.append(point_a)
        # 计算这些点的中心坐标
        center = tuple(map(lambda x: sum(x) / len(x), zip(*points_in_radius)))
        # center = np.mean(np.array(points_in_radius), axis=0)
        center_points.append(center)
    center_points = np.array(center_points, dtype=np.int32)
    return center_points

# 计算切线的向量值
def calculate_normals(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    # 计算法向量（二维情况下）
    normals = np.array([-dy, dx]).T
    # 归一化法向量
    norms = np.linalg.norm(normals, axis=1)
    normalized_normals = normals / norms[:, None]
    return normalized_normals

# 平均法向量
def calculate_moving_average_normals(normals, window_size):
    # 计算滑动窗口的平均法向量
    averaged_normals = np.zeros_like(normals)
    half_window = window_size // 2
    for i in range(len(normals)):
        start = max(0, i - half_window)
        end = min(len(normals), i + half_window + 1)
        averaged_normals[i] = np.mean(normals[start:end], axis=0)
    # 归一化平均法向量
    norms = np.linalg.norm(averaged_normals, axis=1)
    valid_indices = np.where((norms != 0.0) & (~np.isnan(norms)))[0]
    #non_zero_indices = norms != 0
    averaged_normals[valid_indices] /= norms[valid_indices, None]
    return averaged_normals

# 计算角度差
def find_key_points_from_averaged_normals(averaged_normals, threshold):
    dot_products = np.einsum('ij,ij->i', averaged_normals[:-1], averaged_normals[1:])
    dot_products = np.clip(dot_products, -1.0, 1.0)
    angles = np.arccos(dot_products)
    # 确定关键点
    key_point_indices = np.where(angles > threshold)[0]
    key_point_indices = np.append([0], key_point_indices)
    return key_point_indices

def normal_vector(points):
    x = np.array([point[0] for point in points])
    y = np.array([point[1] for point in points])
    window_size = 3  # 窗口大小可以根据需要调整
    normalized_normals = calculate_normals(x, y)
    averaged_normals = calculate_moving_average_normals(normalized_normals, window_size)
    # 设置阈值并找到关键点
    threshold = np.radians(2)  # 例如：10度的阈值
    key_points_indices = find_key_points_from_averaged_normals(averaged_normals, threshold)
    key_points = np.array([x[key_points_indices], y[key_points_indices]]).T
    return key_points

# 边缘点匹配
def match_dilated_boundary(boundary, ori_edge_points, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_boundary = cv2.dilate(boundary, kernel, iterations=1)
    edge_points = np.array(ori_edge_points)
    valid_mask = (edge_points[:, 0] >= 0) & (edge_points[:, 0] < dilated_boundary.shape[0]) & \
                 (edge_points[:, 1] >= 0) & (edge_points[:, 1] < dilated_boundary.shape[1]) & \
                 (dilated_boundary[edge_points[:, 0], edge_points[:, 1]] > 0)
    # 将满足条件的点添加到 obj_edge_points 中
    obj_edge_points = edge_points[valid_mask].tolist()
    obj_edge_points = [(y, x) for x, y in obj_edge_points]            # 逆转坐标
    if len(obj_edge_points) == 0:
        return
    obj_edge_points = np.array(obj_edge_points)
    return obj_edge_points

def get_mask(img_gray, points):
    mask = np.zeros_like(img_gray, dtype=np.uint8)  # 新的mask
    cv2.drawContours(mask, [points], 0, (255, 255, 255), thickness=cv2.FILLED)  # 画出mask
    mask = Image.fromarray(mask, mode='L')
    mask_array = np.array(mask)
    return mask_array

def curve_fitting_mask(img_gray, contour_all, ori_edge_points, k, s, node):
    masks_array = []
    for contour in contour_all:
        mask_edge_points = np.array([point[0] for point in contour])
        inter_mask_edge_points = interpolate_mask_edge_points(mask_edge_points)
        sample = 5
        kernel_size = 10
        radius = 10
        inter_mask_edge_points_nums = len(inter_mask_edge_points)
        node2 = int(inter_mask_edge_points_nums * 0.2)
        if inter_mask_edge_points_nums > k * sample:
            inter_mask_edge_points = inter_mask_edge_points[::sample]  # 采样
        fit_points = b_spline(inter_mask_edge_points, k, s, node)
        if fit_points is None:
            continue
        fit_points2 = b_spline(fit_points, k, s, node2)  # 二次b样条
        if fit_points2 is None or len(fit_points2) < 3:
            continue
        corner_points = normal_vector(fit_points2)  # 自适应采样

        mask_boundary = np.zeros_like(img_gray, dtype=np.uint8)  # 拟合曲线
        cv2.polylines(mask_boundary, [fit_points2], isClosed=True, color=(255, 255, 255), thickness=2)
        obj_edge_points = match_dilated_boundary(mask_boundary, ori_edge_points, kernel_size)
        if obj_edge_points is None:
            continue
        center_points = find_center_points(corner_points, obj_edge_points, radius)
        mask_array = get_mask(img_gray, center_points)
        masks_array.append(mask_array)
    return masks_array

# 将所有mask画在一张图上
def make_color_masks(image, masks_array):
    image_PLI = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    image_PLI = Image.fromarray(image_PLI)
    composite_image = Image.new("RGBA", image_PLI.size)
    for mask_array in masks_array:
        random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 128)
        mask_color_image = Image.new("RGBA", image_PLI.size, random_color)
        mask_img = Image.fromarray(mask_array, mode='L')
        composite_image.paste(mask_color_image, (0, 0), mask_img)
    final_image = Image.alpha_composite(image_PLI, composite_image)
    result_array = np.array(final_image)
    for mask_array in masks_array:
        _, binary_mask = cv2.threshold(mask_array, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result_array, contours, -1, (0, 0, 255, 255), 2)
    result_image = cv2.cvtColor(result_array, cv2.COLOR_BGR2RGB)
    return result_image


def plot_result(img, ann, k=3, s=100, node=200):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度处理
    ori_edge_points = get_canny_points(img_gray)
    contour_all = get_contours(ann)
    masks_array = curve_fitting_mask(img_gray, contour_all, ori_edge_points, k, s, node)
    result_image = make_color_masks(img, masks_array)
    return result_image