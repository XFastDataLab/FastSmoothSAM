# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Run prediction on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ yolo mode=predict model=yolov8n.pt source=0                               # webcam
                                                img.jpg                         # image
                                                vid.mp4                         # video
                                                screen                          # screenshot
                                                path/                           # directory
                                                list.txt                        # list of images
                                                list.streams                    # list of streams
                                                'path/*.jpg'                    # glob
                                                'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ yolo mode=predict model=yolov8n.pt                 # PyTorch
                              yolov8n.torchscript        # TorchScript
                              yolov8n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                              yolov8n_openvino_model     # OpenVINO
                              yolov8n.engine             # TensorRT
                              yolov8n.mlmodel            # CoreML (macOS-only)
                              yolov8n_saved_model        # TensorFlow SavedModel
                              yolov8n.pb                 # TensorFlow GraphDef
                              yolov8n.tflite             # TensorFlow Lite
                              yolov8n_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov8n_paddle_model       # PaddlePaddle
"""
import platform
from pathlib import Path

import cv2
import numpy as np
import torch

from ...nn.autobackend import AutoBackend
from ...yolo.cfg import get_cfg
from ...yolo.data import load_inference_source
from ...yolo.data.augment import LetterBox, classify_transforms
from ...yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, colorstr, ops
from ...yolo.utils.checks import check_imgsz, check_imshow
from ...yolo.utils.files import increment_path
from ...yolo.utils.torch_utils import select_device, smart_inference_mode
import cv2
from skimage.feature import canny
from skimage.color import rgb2gray
from scipy.spatial import KDTree
from scipy.interpolate import splprep, splev
from PIL import Image
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
def linear_interpolation(point1, point2, n):
    x1, y1 = point1
    x2, y2 = point2
    step_size = 1 / n
    interpolated_points = [(x1 + i * step_size * (x2 - x1), y1 + i * step_size * (y2 - y1)) for i in range(n + 1)]
    return interpolated_points
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
def unique_datas(points_data):
    unique_data, unique_indices = np.unique(points_data, axis=0, return_index=True)        # 去重
    unique_data_ordered = unique_data[np.argsort(unique_indices)]                          # 按原序排序
    return unique_data_ordered
def unique_data(points):
    # 此函数用于去重点集，你可以根据你的具体需求实现
    return np.unique(points, axis=0)
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
def BS(ori_mask,ori_img):
    ori_edge_points = skimage_canny(ori_img, 1)  # canny获取边缘点
    contour_all, _ = get_contours(ori_mask)  # 获取所有contour图层

    # masks_array =mask_all
    masks_array = []
    for contour in contour_all:
        mask_edge_points = np.array([point[0] for point in contour])  # 提取边缘点
        inter_mask_edge_points = interpolate_mask_edge_points(mask_edge_points)  # 补充边缘点
        k1 = 3  # 一次B样条阶数
        kernel_size = 5  # 膨胀核大小
        inter_mask_edge_points_nums = len(inter_mask_edge_points)
        node1 = int(inter_mask_edge_points_nums * 0.6)  # 一次拟合生成点的数量
        fit_points, x_fit, y_fit, curvatures = b_spline(inter_mask_edge_points, k1, node1, 1)
        if fit_points is None or len(fit_points) < 3:
            continue

        fit_points_int = np.array(fit_points, dtype=np.int32)
        img_gray = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)  # 灰度处理
        new_mask = np.zeros_like(img_gray, dtype=np.uint8)  # 新的mask

        mask_boundary = np.zeros_like(img_gray, dtype=np.uint8)  # 拟合曲线
        cv2.polylines(mask_boundary, [fit_points_int], isClosed=True, color=(255, 255, 255), thickness=2)
        obj_edge_points = match_dilated_boundary(mask_boundary, ori_edge_points, kernel_size)  # 膨胀曲线，得到在曲线中的canny数据点

        if obj_edge_points is None:
            mask_array = draw_mask(new_mask, fit_points_int)
            masks_array.append(mask_array)
            continue

        corner_points = AdS(fit_points, curvatures, 0.02)  # 自适应采样，得到高曲率的点
        merged_points_array = np.concatenate((fit_points, obj_edge_points))
        thin_points = get_thin_points(merged_points_array, fit_points, corner_points, 5)  # 打薄

        k2 = 1  # 二次拟合的阶数
        node2 = len(thin_points)  # 二次拟合生成的点的数量
        if node2 < 3:
            mask_array = draw_mask(new_mask, fit_points)
            masks_array.append(mask_array)
            continue
        fit_points2, _, _, _ = b_spline(thin_points, k2, node2, 2)  # 二次b样条
        if fit_points2 is None:
            mask_array = draw_mask(new_mask, fit_points_int)  # 二次拟合失败，以一次拟合的结果为最终结果
            masks_array.append(mask_array)
            continue
        fit_points2_int = np.array(fit_points2, dtype=np.int32)
        cv2.polylines(ori_img, [fit_points2_int], isClosed=True, color=(255, 255, 0), thickness=2)
        new_mask = np.zeros_like(img_gray, dtype=np.uint8)  # 新的mask
        cv2.drawContours(new_mask, [fit_points2_int], 0, (255, 255, 255), thickness=cv2.FILLED)  # 画出mask
        mask_array = draw_mask(new_mask, fit_points_int)
        masks_array.append(mask_array)
    return masks_array
STREAM_WARNING = """
    WARNING ⚠️ stream/video/webcam/dir predict source will accumulate results in RAM unless `stream=True` is passed,
    causing potential out-of-memory errors for large sources or long-running streams/videos.

    Usage:
        results = model(source=..., stream=True)  # generator of Results objects
        for r in results:
            boxes = r.boxes  # Boxes object for bbox outputs
            masks = r.masks  # Masks object for segment masks outputs
            probs = r.probs  # Class probabilities for classification outputs
"""


class BasePredictor:
    """
    BasePredictor

    A base class for creating predictors.

    Attributes:
        args (SimpleNamespace): Configuration for the predictor.
        save_dir (Path): Directory to save results.
        done_setup (bool): Whether the predictor has finished setup.
        model (nn.Module): Model used for prediction.
        data (dict): Data configuration.
        device (torch.device): Device used for prediction.
        dataset (Dataset): Dataset used for prediction.
        vid_path (str): Path to video file.
        vid_writer (cv2.VideoWriter): Video writer for saving video output.
        annotator (Annotator): Annotator used for prediction.
        data_path (str): Path to data.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the BasePredictor class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
        name = self.args.name or f'{self.args.mode}'
        self.save_dir = increment_path(Path(project) / name, exist_ok=self.args.exist_ok)
        if self.args.conf is None:
            self.args.conf = 0.25  # default conf=0.25
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # Usable if setup is done
        self.model = None
        self.data = self.args.data  # data_dict
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_path, self.vid_writer = None, None
        self.plotted_img = None
        self.data_path = None
        self.source_type = None
        self.batch = None
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        callbacks.add_integration_callbacks(self)

    def preprocess(self, im):
        """Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.
        """
        if not isinstance(im, torch.Tensor):
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)
        # NOTE: assuming im with (b, 3, h, w) if it's a tensor
        img = im.to(self.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def pre_transform(self, im):
        """Pre-tranform input image before inference.

        Args:
            im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Return: A list of transformed imgs.
        """
        same_shapes = all(x.shape == im[0].shape for x in im)
        auto = same_shapes and self.model.pt
        return [LetterBox(self.imgsz, auto=auto, stride=self.model.stride)(image=x) for x in im]

    def write_results(self, idx, results, batch):
        """Write inference results to a file or directory."""
        p, im, _ = batch
        log_string = ''
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        if self.source_type.webcam or self.source_type.from_img:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        result = results[idx]
        log_string += result.verbose()

        if self.args.save or self.args.show:  # Add bbox to image
            plot_args = dict(line_width=self.args.line_width,
                             boxes=self.args.boxes,
                             conf=self.args.show_conf,
                             labels=self.args.show_labels)
            if not self.args.retina_masks:
                plot_args['im_gpu'] = im[idx]
            self.plotted_img = result.plot(**plot_args)
        # Write
        if self.args.save_txt:
            result.save_txt(f'{self.txt_path}.txt', save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / 'crops', file_name=self.data_path.stem)

        return log_string

    def postprocess(self, preds, img, orig_img):
        """Post-processes predictions for an image and returns them."""
        return preds

    def __call__(self, source=None, model=None, stream=False):
        """Performs inference on an image or stream."""
        self.stream = stream
        if stream:
            return self.stream_inference(source, model)
        else:
            return list(self.stream_inference(source, model))  # merge list of Result into one

    def predict_cli(self, source=None, model=None):
        """Method used for CLI prediction. It uses always generator as outputs as not required by CLI mode."""
        gen = self.stream_inference(source, model)
        for _ in gen:  # running CLI inference without accumulating any outputs (do not modify)
            pass

    def setup_source(self, source):
        """Sets up source and inference mode."""
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # check image size
        self.transforms = getattr(self.model.model, 'transforms', classify_transforms(
            self.imgsz[0])) if self.args.task == 'classify' else None
        self.dataset = load_inference_source(source=source, imgsz=self.imgsz, vid_stride=self.args.vid_stride)
        self.source_type = self.dataset.source_type
        if not getattr(self, 'stream', True) and (self.dataset.mode == 'stream' or  # streams
                                                  len(self.dataset) > 1000 or  # images
                                                  any(getattr(self.dataset, 'video_flag', [False]))):  # videos
            LOGGER.warning(STREAM_WARNING)
        self.vid_path, self.vid_writer = [None] * self.dataset.bs, [None] * self.dataset.bs

    @smart_inference_mode()
    def stream_inference(self, source=None, model=None):
        """Streams real-time inference on camera feed and saves results to file."""
        if self.args.verbose:
            LOGGER.info('')

        # Setup model
        if not self.model:
            self.setup_model(model)
        # Setup source every time predict is called
        self.setup_source(source if source is not None else self.args.source)

        # Check if save_dir/ label file exists
        if self.args.save or self.args.save_txt:
            (self.save_dir / 'labels' if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        # Warmup model
        if not self.done_warmup:
            self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
            self.done_warmup = True

        self.seen, self.windows, self.batch, profilers = 0, [], None, (ops.Profile(), ops.Profile(), ops.Profile())
        self.run_callbacks('on_predict_start')
        for batch in self.dataset:
            self.run_callbacks('on_predict_batch_start')
            self.batch = batch
            path, im0s, vid_cap, s = batch
            visualize = increment_path(self.save_dir / Path(path[0]).stem,
                                       mkdir=True) if self.args.visualize and (not self.source_type.tensor) else False

            # Preprocess
            with profilers[0]:
                im = self.preprocess(im0s)
            # Inference
            with profilers[1]:
                preds = self.model(im, augment=self.args.augment, visualize=visualize)
            # Postprocess
            with profilers[2]:
                self.results = self.postprocess(preds, im, im0s)
                new_result=self.results[0].masks.data
                new_B_result=BS(new_result,np.array(im0s[0]))
                self.results[0].masks.data=new_B_result
            self.run_callbacks('on_predict_postprocess_end')

            # Visualize, save, write results
            n = len(im0s)
            for i in range(n):
                self.results[i].speed = {
                    'preprocess': profilers[0].dt * 1E3 / n,
                    'inference': profilers[1].dt * 1E3 / n,
                    'postprocess': profilers[2].dt * 1E3 / n}
                if self.source_type.tensor:  # skip write, show and plot operations if input is raw tensor
                    continue
                p, im0 = path[i], im0s[i].copy()
                p = Path(p)

                if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                    s += self.write_results(i, self.results, (p, im, im0))

                if self.args.show and self.plotted_img is not None:
                    self.show(p)

                if self.args.save and self.plotted_img is not None:
                    self.save_preds(vid_cap, i, str(self.save_dir / p.name))
            self.run_callbacks('on_predict_batch_end')
            yield from self.results

            # Print time (inference-only)
            if self.args.verbose:
                LOGGER.info(f'{s}{profilers[1].dt * 1E3:.1f}ms')

        # Release assets
        if isinstance(self.vid_writer[-1], cv2.VideoWriter):
            self.vid_writer[-1].release()  # release final video writer

        # Print results
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1E3 for x in profilers)  # speeds per image
            LOGGER.info(f'Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape '
                        f'{(1, 3, *self.imgsz)}' % t)
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob('labels/*.txt')))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")

        self.run_callbacks('on_predict_end')

    def setup_model(self, model, verbose=True):
        """Initialize YOLO model with given parameters and set it to evaluation mode."""
        device = select_device(self.args.device, verbose=verbose)
        model = model or self.args.model
        self.args.half &= device.type != 'cpu'  # half precision only supported on CUDA
        self.model = AutoBackend(model,
                                 device=device,
                                 dnn=self.args.dnn,
                                 data=self.args.data,
                                 fp16=self.args.half,
                                 fuse=True,
                                 verbose=verbose)
        self.device = device
        self.model.eval()

    def show(self, p):
        """Display an image in a window using OpenCV imshow()."""
        im0 = self.plotted_img
        if platform.system() == 'Linux' and p not in self.windows:
            self.windows.append(p)
            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
        cv2.imshow(str(p), im0)
        cv2.waitKey(500 if self.batch[3].startswith('image') else 1)  # 1 millisecond

    def save_preds(self, vid_cap, idx, save_path):
        """Save video predictions as mp4 at specified path."""
        im0 = self.plotted_img
        # Save imgs
        if self.dataset.mode == 'image':
            cv2.imwrite(save_path, im0)
        else:  # 'video' or 'stream'
            if self.vid_path[idx] != save_path:  # new video
                self.vid_path[idx] = save_path
                if isinstance(self.vid_writer[idx], cv2.VideoWriter):
                    self.vid_writer[idx].release()  # release previous video writer
                if vid_cap:  # video
                    fps = int(vid_cap.get(cv2.CAP_PROP_FPS))  # integer required, floats produce error in MP4 codec
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                self.vid_writer[idx] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            self.vid_writer[idx].write(im0)

    def run_callbacks(self, event: str):
        """Runs all registered callbacks for a specific event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def add_callback(self, event: str, func):
        """
        Add callback
        """
        self.callbacks[event].append(func)
