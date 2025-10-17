# --- START OF FILE distance_cal_by_depth.py (ENHANCED WITH DEPTH PROBE) ---

import cv2
import numpy as np
import argparse
import os

# 用于存储用户点击的点 (存储的是在原始高分辨率图像上的坐标)
selected_points = []
# 用于传递给回调函数的参数
callback_params = {}

def pixel_to_3d(u, v, depth_map, K):
    """
    将单个像素坐标(u,v)反向投影到三维空间坐标(X,Y,Z).
    这里的 u, v, depth_map, K 都必须是在同一个尺度下的。
    """
    if not (0 <= v < depth_map.shape[0] and 0 <= u < depth_map.shape[1]):
        print(f"Warning: Coordinate ({u}, {v}) is out of depth map bounds ({depth_map.shape[1]}, {depth_map.shape[0]}).")
        return None

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    Z = depth_map[v, u]
    
    if Z <= 0 or np.isinf(Z) or np.isnan(Z):
        return None
        
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.array([X, Y, Z])

# --- 新增辅助函数 ---
def show_depth_at_point(point_2d_orig, point_index):
    """
    查询并打印单个点的深度信息.
    """
    depth_map = callback_params['depth_map']
    depth_map_scale = callback_params['depth_map_scale']

    # 将原始图像坐标缩放到深度图的坐标系
    u = int(point_2d_orig[0] * depth_map_scale)
    v = int(point_2d_orig[1] * depth_map_scale)

    # 检查坐标是否在深度图范围内
    if not (0 <= v < depth_map.shape[0] and 0 <= u < depth_map.shape[1]):
        depth_info = "out of bounds"
    else:
        depth_value = depth_map[v, u]
        if depth_value <= 0 or np.isinf(depth_value) or np.isnan(depth_value):
            depth_info = "invalid"
        else:
            depth_info = f"{depth_value:.3f} meters"
    
    print(f" -> Depth at Point {point_index}: {depth_info}")


def mouse_callback(event, x, y, flags, param):
    """
    OpenCV鼠标事件的回调函数.
    """
    global selected_points, callback_params

    display_scale = callback_params['display_scale']
    depth_map_scale = callback_params['depth_map_scale']

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(selected_points) == 2:
            selected_points.clear()
            callback_params['display_image'] = callback_params['original_display_image'].copy()

        original_x = int(x / display_scale)
        original_y = int(y / display_scale)
        current_point = (original_x, original_y)
        selected_points.append(current_point)
        
        print(f"\nPoint {len(selected_points)} selected at original pixel coords: ({original_x}, {original_y})")
        cv2.circle(callback_params['display_image'], (x, y), 5, (0, 255, 0), -1)

        # --- 功能增强: 每次点击都显示深度 ---
        show_depth_at_point(current_point, len(selected_points))
        # -----------------------------------

        if len(selected_points) == 2:
            p1_2d_orig = selected_points[0]
            p2_2d_orig = selected_points[1]

            u1 = int(p1_2d_orig[0] * depth_map_scale)
            v1 = int(p1_2d_orig[1] * depth_map_scale)
            u2 = int(p2_2d_orig[0] * depth_map_scale)
            v2 = int(p2_2d_orig[1] * depth_map_scale)

            p1_3d = pixel_to_3d(u1, v1, callback_params['depth_map'], callback_params['K_scaled'])
            p2_3d = pixel_to_3d(u2, v2, callback_params['depth_map'], callback_params['K_scaled'])

            if p1_3d is not None and p2_3d is not None:
                distance = np.linalg.norm(p1_3d - p2_3d)
                
                p1_display_coords = (int(p1_2d_orig[0] * display_scale), int(p1_2d_orig[1] * display_scale))
                p2_display_coords = (x, y) # (x, y) 就是第二次点击的显示坐标
                cv2.line(callback_params['display_image'], p1_display_coords, p2_display_coords, (0, 0, 255), 2)
                
                print("\n--- Measurement Result ---")
                print(f"Point 1 (3D): [{p1_3d[0]:.3f}, {p1_3d[1]:.3f}, {p1_3d[2]:.3f}] meters")
                print(f"Point 2 (3D): [{p2_3d[0]:.3f}, {p2_3d[1]:.3f}, {p2_3d[2]:.3f}] meters")
                print(f"Distance between points: {distance:.4f} meters")
                print("--------------------------\n")
            else:
                print("\nError: Could not calculate distance. One or both selected points have invalid depth.\n")
            
            print("Click again to start a new measurement.")

def main(args):
    global callback_params

    if not all(os.path.exists(f) for f in [args.depth_file, args.intrinsic_file, args.image_file]):
        print("Error: One or more input files not found. Please check paths.")
        return

    print("Loading data...")
    depth_map = np.load(args.depth_file)
    image = cv2.imread(args.image_file)
    with open(args.intrinsic_file, 'r') as f:
        lines = f.readlines()
        K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3, 3)

    expected_h = int(image.shape[0] * args.depth_map_scale)
    expected_w = int(image.shape[1] * args.depth_map_scale)
    if depth_map.shape[0] != expected_h or depth_map.shape[1] != expected_w:
        print(f"Warning: Depth map dimensions do not match the expected size.")
        print("Please ensure --depth_map_scale is set correctly.")
        
    K_scaled = K.copy()
    K_scaled[:2, :] *= args.depth_map_scale
    print("\nOriginal Intrinsics (K):\n", K)
    print(f"\nScaled Intrinsics (K_scaled) with scale={args.depth_map_scale}:\n", K_scaled)

    display_scale = args.display_scale
    display_width = int(image.shape[1] * display_scale)
    display_height = int(image.shape[0] * display_scale)
    display_image = cv2.resize(image, (display_width, display_height), interpolation=cv2.INTER_AREA)

    callback_params = {
        'display_image': display_image.copy(),
        'original_display_image': display_image.copy(),
        'depth_map': depth_map,
        'K_scaled': K_scaled,
        'display_scale': display_scale,
        'depth_map_scale': args.depth_map_scale
    }
    
    window_name = "Distance & Depth Calculator"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_width, display_height)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n--- Instructions ---")
    print("1. Click a point to check its depth.")
    print("2. Click a second point to measure the distance between them.")
    print("3. Press 'r' to reset, 'q' to quit.")
    print("--------------------")

    while True:
        cv2.imshow(window_name, callback_params['display_image'])
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            selected_points.clear()
            callback_params['display_image'] = callback_params['original_display_image'].copy()
            print("\nSelection reset. Please select a new point.")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate 3D distance and query depth from a depth map.")
    
    parser.add_argument('--depth_file', type=str, default='output_usb/depth_meter.npy', 
                        help='Path to the .npy depth map file.')
    
    parser.add_argument('--intrinsic_file', type=str, required=True, 
                        help='Path to the camera intrinsic matrix file.')
                        
    parser.add_argument('--image_file', type=str, required=True,
                        help='Path to the corresponding original high-resolution left image for display.')
    
    parser.add_argument('--display_scale', type=float, default=1,
                        help='Scale factor for the display window.')

    parser.add_argument('--depth_map_scale', type=float, default=1,
                        help='The scale factor used to generate the depth map. Must match the --scale used in run_demo.py.')

    args = parser.parse_args()
    main(args)