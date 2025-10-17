# --- START OF OPTIMIZED stereo_rectification.py ---
import cv2
import numpy as np
import logging
from pathlib import Path
import argparse
from typing import Tuple, List, Dict, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def parse_dat_file(filepath: Path) -> Dict[str, np.ndarray]:
    """解析.dat文件，提取矩阵数据（K, D, R, T等）"""
    try:
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
    except IOError as e:
        logger.error(f"读取文件 {filepath} 失败: {e}")
        raise

    data: Dict[str, np.ndarray] = {}
    current_key: Optional[str] = None
    matrix_data: List[List[float]] = []

    for line in lines:
        if line.endswith(':'):
            # 保存上一个矩阵
            if current_key and matrix_data:
                data[current_key] = np.array(matrix_data, dtype=np.float32)
            current_key = line[:-1].lower()
            matrix_data = []
        else:
            try:
                matrix_data.append([float(x) for x in line.split()])
            except ValueError:
                logger.warning(f"文件 {filepath} 中行 '{line}' 包含非数值数据，已跳过")

    # 处理最后一个矩阵
    if current_key and matrix_data:
        data[current_key] = np.array(matrix_data, dtype=np.float32)
    
    return data


def load_calibration_parameters(params_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """加载相机标定参数（内参、畸变系数、外参R/T）并验证合法性"""
    required_files = [
        'camera0_intrinsics.dat',
        'camera1_intrinsics.dat',
        'camera1_rot_trans.dat'
    ]
    
    # 检查文件是否存在
    for file in required_files:
        if not (params_dir / file).exists():
            raise FileNotFoundError(f"标定文件缺失: {params_dir / file}")

    # 解析文件
    cam0_intr = parse_dat_file(params_dir / 'camera0_intrinsics.dat')
    cam1_intr = parse_dat_file(params_dir / 'camera1_intrinsics.dat')
    cam1_extr = parse_dat_file(params_dir / 'camera1_rot_trans.dat')

    # 提取参数并验证维度
    try:
        K_left = cam0_intr['intrinsic']
        D_left = cam0_intr['distortion'].flatten()  # 确保为1D数组
        K_right = cam1_intr['intrinsic']
        D_right = cam1_intr['distortion'].flatten()
        R = cam1_extr['r']
        T = cam1_extr['t'].flatten()  # 确保为1D数组（3x1）
    except KeyError as e:
        raise ValueError(f"标定文件格式错误，缺失关键参数: {e}")

    # 验证矩阵维度
    if K_left.shape != (3, 3) or K_right.shape != (3, 3):
        raise ValueError("内参矩阵K必须为3x3矩阵")
    if R.shape != (3, 3):
        raise ValueError("旋转矩阵R必须为3x3矩阵")
    if T.shape != (3,):
        raise ValueError("平移向量T必须为3元素数组（3x1）")
    if D_left.size not in (5, 8) or D_right.size not in (5, 8):
        raise ValueError("畸变系数必须为5参数（k1,k2,p1,p2,k3）或8参数模型")

    logger.info("标定参数加载并验证成功")
    return K_left, D_left, K_right, D_right, R, T


def find_and_pair_images(base_dir: Path, ext: str) -> Tuple[List[Path], List[Path]]:
    """匹配左右相机的图像对（基于文件名后缀标识符）"""
    left_dir = base_dir / 'left'
    right_dir = base_dir / 'right'

    # 检查目录是否存在
    if not left_dir.is_dir() or not right_dir.is_dir():
        raise NotADirectoryError(f"图像子目录不存在: left={left_dir.exists()}, right={right_dir.exists()}")

    # 获取所有图像路径
    left_paths = sorted(left_dir.glob(f'*.{ext}'))
    right_paths = sorted(right_dir.glob(f'*.{ext}'))

    if not left_paths:
        raise FileNotFoundError(f"未在 {left_dir} 找到 {ext} 格式图像")
    if not right_paths:
        raise FileNotFoundError(f"未在 {right_dir} 找到 {ext} 格式图像")

    logger.info(f"找到左图 {len(left_paths)} 张，右图 {len(right_paths)} 张")

    # 基于文件名后缀标识符匹配（如"img_001.jpg"中的"001"）
    left_map = {}
    for path in left_paths:
        try:
            # 提取文件名中最后一个"_"后的数字（不含扩展名）
            identifier = path.stem.split('_')[-1]
            left_map[identifier] = path
        except IndexError:
            logger.warning(f"文件名格式异常，跳过左图: {path.name}")

    paired_left: List[Path] = []
    paired_right: List[Path] = []
    for right_path in right_paths:
        try:
            identifier = right_path.stem.split('_')[-1]
            if identifier in left_map:
                paired_left.append(left_map[identifier])
                paired_right.append(right_path)
        except IndexError:
            logger.warning(f"文件名格式异常，跳过右图: {right_path.name}")

    if not paired_left:
        raise ValueError("未匹配到任何图像对，请检查文件名格式是否一致")
    
    # 检查匹配率
    match_ratio = len(paired_left) / max(len(left_paths), len(right_paths))
    if match_ratio < 0.5:
        logger.warning(f"图像对匹配率较低({match_ratio:.2%})，可能存在文件名格式问题")

    logger.info(f"成功匹配 {len(paired_left)} 对图像")
    return paired_left, paired_right


def rectify_images(
    img_left: np.ndarray,
    img_right: np.ndarray,
    K_left: np.ndarray,
    D_left: np.ndarray,
    K_right: np.ndarray,
    D_right: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    alpha: float = -1  # 新增alpha参数控制裁剪方式
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """执行立体校正，返回校正后的图像和左相机投影矩阵P_left"""
    # 检查图像尺寸和通道数
    if img_left.shape[:2] != img_right.shape[:2]:
        raise ValueError(f"左右图像尺寸不匹配: 左图{img_left.shape[:2]}, 右图{img_right.shape[:2]}")
    if img_left.shape[2:] != img_right.shape[2:]:
        raise ValueError(f"左右图像通道数不匹配: 左图{img_left.shape[2:]}, 右图{img_right.shape[2:]}")
        
    height, width = img_left.shape[:2]
    image_size = (width, height)

    # 立体校正计算
    try:
        R_left, R_right, P_left, P_right, Q, _, _ = cv2.stereoRectify(
            K_left, D_left, K_right, D_right, image_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha
        )
    except cv2.error as e:
        raise RuntimeError(f"立体校正计算失败: {e}")

    # 计算畸变校正和立体校正的映射表
    map_left_x, map_left_y = cv2.initUndistortRectifyMap(
        K_left, D_left, R_left, P_left, image_size, cv2.CV_32FC1
    )
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(
        K_right, D_right, R_right, P_right, image_size, cv2.CV_32FC1
    )

    # 应用映射表得到校正图像
    rectified_left = cv2.remap(img_left, map_left_x, map_left_y, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(img_right, map_right_x, map_right_y, cv2.INTER_LINEAR)

    # 检查校正结果有效性
    if rectified_left.size == 0 or rectified_right.size == 0:
        raise RuntimeError("校正后图像为空，可能是映射表计算错误")

    return rectified_left, rectified_right, P_left


def calculate_epipolar_error(rect_left, rect_right, checkerboard_size):
    """
    计算校正后图像对的极线偏差（基于棋盘格角点）
    参数:
        checkerboard_size: 棋盘格角点数量 (rows, cols)
    返回：平均偏差、最大偏差、角点数量
    """
    checkerboard_rows, checkerboard_cols = checkerboard_size
    # 转换为灰度图（显式指定读取模式避免通道问题）
    if len(rect_left.shape) == 3:
        gray_left = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
    else:
        gray_left = rect_left.copy()
        
    if len(rect_right.shape) == 3:
        gray_right = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)
    else:
        gray_right = rect_right.copy()
    
    # 检测棋盘格角点
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, (checkerboard_cols, checkerboard_rows), None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, (checkerboard_cols, checkerboard_rows), None)
    
    if not ret_left or not ret_right:
        logger.warning("未检测到棋盘格角点，无法计算极线偏差")
        return None, None, 0
    
    # 亚像素级角点优化
    corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
    corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
    
    # 确保角点数量一致
    if len(corners_left) != len(corners_right):
        logger.warning(f"左右图像角点数量不一致({len(corners_left)} vs {len(corners_right)})，无法计算极线偏差")
        return None, None, 0
    
    # 计算每个对应角点的y坐标差（极线偏差）
    errors = []
    for (x1, y1), (x2, y2) in zip(corners_left.reshape(-1, 2), corners_right.reshape(-1, 2)):
        errors.append(abs(y1 - y2))  # 极线理想情况下y坐标应相同
    
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    logger.info(f"极线偏差 - 平均: {mean_error:.4f}px, 最大: {max_error:.4f}px, 角点数量: {len(errors)}")
    return mean_error, max_error, len(errors)


def visualize_rectification(
    rect_left: np.ndarray,
    rect_right: np.ndarray,
    left_name: str,
    right_name: str,
    baseline: float,
    save_path: Optional[Path] = None
) -> None:
    """可视化校正结果，显示水平参考线、图像名称和基线信息，支持保存"""
    # 拼接左右图像
    combined = np.hstack((rect_left, rect_right))
    h, w = combined.shape[:2]

    # 缩放图像以适应显示（最大宽度1800）
    max_display_width = 1800
    if w > max_display_width:
        scale = max_display_width / w
        combined = cv2.resize(combined, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = combined.shape[:2]

    # 绘制水平参考线（15条）
    for i in range(1, 15):
        y = int(h * i / 15)
        cv2.line(combined, (0, y), (w, y), (0, 255, 0), 1)  # 绿色水平线

    # 添加文本信息
    cv2.putText(
        combined, f"Left: {left_name}", (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
    )
    cv2.putText(
        combined, f"Right: {right_name}", (w//2 + 20, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
    )
    cv2.putText(
        combined, f"Baseline: {baseline:.4f} m", (20, h-20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2
    )

    # 显示并等待按键
    cv2.imshow("Rectification Check (Press 's' to save, any key to continue)", combined)
    key = cv2.waitKey(0)
    if key == ord('s') and save_path:
        cv2.imwrite(str(save_path), combined)
        logger.info(f"可视化结果已保存至: {save_path}")
    cv2.destroyAllWindows()


def save_rectified_results(
    rect_left: np.ndarray,
    rect_right: np.ndarray,
    left_path: Path,
    right_path: Path,
    output_dir: Path
) -> None:
    """保存校正后的图像到输出目录"""
    output_left = output_dir / 'left' / left_path.name
    output_right = output_dir / 'right' / right_path.name
    # 检查保存结果
    if not cv2.imwrite(str(output_left), rect_left):
        logger.error(f"无法保存校正图像: {output_left}")
    if not cv2.imwrite(str(output_right), rect_right):
        logger.error(f"无法保存校正图像: {output_right}")
    logger.debug(f"已保存校正图像: {output_left.name} 和 {output_right.name}")


def save_calibration_output(K_new: np.ndarray, baseline: float, output_dir: Path) -> None:
    """保存校正后的内参K'和基线到HiK.txt"""
    # 验证K_new有效性
    if K_new.shape != (3, 3):
        logger.error(f"无效的内参矩阵K'，形状应为(3,3)，实际为{K_new.shape}")
        return
        
    k_txt_path = output_dir / 'HiK.txt'
    try:
        with open(k_txt_path, 'w') as f:
            f.write(' '.join(map(lambda x: f"{x:.6f}", K_new.flatten())) + '\n')  # 保留6位小数
            f.write(f"{baseline:.6f}\n")
        logger.info(f"校正参数已保存至: {k_txt_path}")
    except IOError as e:
        logger.error(f"无法保存校正参数至 {k_txt_path}: {e}")


def main(args):
    # 转换为Path对象
    params_dir = Path(args.params_dir)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)

    # 创建输出目录
    (output_dir / 'left').mkdir(parents=True, exist_ok=True)
    (output_dir / 'right').mkdir(parents=True, exist_ok=True)

    try:
        # 1. 加载标定参数
        K_left, D_left, K_right, D_right, R, T_calib = load_calibration_parameters(params_dir)

        # 2. 匹配图像对
        left_paths, right_paths = find_and_pair_images(image_dir, args.image_ext)

        # 3. 处理图像对
        processed_first = False
        for i, (left_path, right_path) in enumerate(zip(left_paths, right_paths)):
            # 读取图像（显式指定色彩模式）
            img_left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
            img_right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
            if img_left is None or img_right is None:
                logger.warning(f"跳过无效图像对: {left_path.name}, {right_path.name}")
                continue

            logger.info(f"处理图像对 ({i+1}/{len(left_paths)}): {left_path.name} & {right_path.name}")

            # 立体校正
            rect_left, rect_right, P_left_new = rectify_images(
                img_left, img_right, K_left, D_left, K_right, D_right, R, T_calib,
                alpha=args.alpha  # 使用命令行参数控制alpha
            )

            # 保存校正图像
            save_rectified_results(rect_left, rect_right, left_path, right_path, output_dir)

            # 处理第一张图：计算基线、生成HiK.txt、可视化、极线偏差
            if not processed_first:
                # 计算基线（转换为米）
                unit_map = {'mm': 1000.0, 'cm': 100.0, 'm': 1.0}
                conversion = unit_map.get(args.baseline_unit.lower(), 1.0)
                baseline_meters = np.linalg.norm(T_calib) / conversion
                logger.info(f"基线计算完成: {baseline_meters:.4f} 米 (原始单位: {args.baseline_unit})")

                # 提取校正后的内参K'
                K_new = P_left_new[:3, :3]

                # 保存HiK.txt
                save_calibration_output(K_new, baseline_meters, output_dir)

                # 可视化校正结果
                viz_save_path = output_dir / 'rectification_visualization.jpg'
                visualize_rectification(
                    rect_left, rect_right,
                    left_path.name, right_path.name,
                    baseline_meters,
                    viz_save_path
                )
                # 计算极线偏差（使用命令行参数传入的棋盘格尺寸）
                calculate_epipolar_error(rect_left, rect_right, args.checkerboard_size)

                processed_first = True

        logger.info("立体校正完成，所有结果已保存")

    except Exception as e:
        logger.error(f"程序执行失败: {e}", exc_info=True)  # 打印详细错误堆栈
        exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="立体图像校正与参数生成工具")
    parser.add_argument(
        '--params_dir', type=str, default='camera_parameters',
        help='标定参数.dat文件所在目录'
    )
    parser.add_argument(
        '--image_dir', type=str, required=True,
        help='包含left和right子目录的图像根目录'
    )
    parser.add_argument(
        '--output_dir', type=str, default='rectified_output_3',
        help='校正后图像和参数的保存目录'
    )
    parser.add_argument(
        '--image_ext', type=str, default='bmp',
        help='图像文件扩展名（如png、jpg）'
    )
    parser.add_argument(
        '--baseline_unit', type=str, default='cm', choices=['mm', 'cm', 'm'],
        help='基线原始单位（与标定T向量一致）'
    )
    parser.add_argument(
        '--checkerboard_size', type=int, nargs=2, default=(8, 8),
        help='棋盘格角点数量 (行数 列数)，如 --checkerboard_size 8 8'
    )
    parser.add_argument(
        '--alpha', type=float, default=0,
        help='立体校正裁剪参数，-1=保留所有像素，0=裁剪无效像素'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='启用调试模式（显示详细日志）'
    )
    args = parser.parse_args()

    # 调试模式切换日志级别
    if args.debug:
        logger.setLevel(logging.DEBUG)

    main(args)
