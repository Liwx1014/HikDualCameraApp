# --- START OF FILE calibrate_from_folder.py (ENHANCED DISPLAY & PRECISION) ---

import cv2 as cv
import glob
import numpy as np
import sys
import yaml
import os
import tkinter as tk  # Import tkinter to get screen resolution
import argparse
# This will contain the calibration settings from the calibration_settings.yaml file
calibration_settings = {}

# --- NEW HELPER FUNCTIONS for better display ---
def get_screen_resolution():
    """Uses tkinter to get the primary screen's resolution."""
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        return width, height
    except Exception as e:
        print(f"Warning: Could not get screen resolution using tkinter ({e}). Falling back to default size.")
        return 1920, 1080 # Fallback to a common resolution

def calculate_optimal_display_size(img_width, img_height, screen_width, screen_height, num_images_h=1):
    """
    Calculates the best display size to fit an image (or multiple images horizontally) on the screen.
    - num_images_h: Number of images to be displayed side-by-side.
    """
    # Leave a margin (e.g., 10%) for window borders and taskbars
    max_display_width = screen_width * 0.9
    max_display_height = screen_height * 0.9

    # The available width for each image
    available_width = max_display_width / num_images_h

    # Calculate the scaling factor based on width and height constraints
    scale_w = available_width / img_width
    scale_h = max_display_height / img_height
    
    # Use the smaller scale factor to maintain aspect ratio and fit within both dimensions
    # Also, ensure we don't scale up small images (max scale of 1.0)
    scale = min(scale_w, scale_h, 1.0)

    return (int(img_width * scale), int(img_height * scale))
# --- END of new helper functions ---


def parse_calibration_settings_file(filename):
    """Open and load the calibration_settings.yaml file"""
    global calibration_settings
    if not os.path.exists(filename):
        print('File does not exist:', filename); quit()
    
    print('Using for calibration settings: ', filename)
    with open(filename) as f:
        calibration_settings = yaml.safe_load(f)

    if 'checkerboard_rows' not in calibration_settings:
        print('Error: "checkerboard_rows" key was not found in the settings file.'); quit()


def calibrate_camera_for_intrinsic_parameters(image_paths, screen_res):
    """
    Calibrate a single camera to find its intrinsic parameters.
    (MODIFIED FOR HIGHER PRECISION AND BETTER DIAGNOSTICS)
    """
    if not image_paths:
        print("Error: No images found for intrinsic calibration."); quit()

    images = [cv.imread(imname, 1) for imname in image_paths]
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale']

    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling * objp

    width, height = images[0].shape[1], images[0].shape[0]
    imgpoints, objpoints = [], []
    used_image_names = [] # Keep track of images that are not skipped
    
    display_size = calculate_optimal_display_size(width, height, screen_res[0], screen_res[1])
    window_name = 'Intrinsic Calibration - Press "s" to skip, any other key to continue'
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, display_size[0], display_size[1])

    print(f"Processing {len(images)} images for intrinsic calibration...")
    for i, frame in enumerate(images):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

        if ret == True:
            corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv.drawChessboardCorners(frame, (rows,columns), corners, ret)
            # 在cv.drawChessboardCorners之后添加角点序号标注
            for j, corner in enumerate(corners):
                cv.putText(frame, f"{j}", (int(corner[0][0]), int(corner[0][1])), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                
            display_frame = cv.resize(frame, display_size)
            cv.imshow(window_name, display_frame)
            k = cv.waitKey(0)

            if k & 0xFF == ord('s'):
                print(f"Skipping image {i+1}/{len(images)}: {os.path.basename(image_paths[i])}"); continue

            objpoints.append(objp)
            imgpoints.append(corners)
            used_image_names.append(os.path.basename(image_paths[i]))

    cv.destroyAllWindows()
    if not objpoints:
        print("Error: No valid checkerboards found. Cannot calibrate."); quit()
    
    # --- MODIFICATION 1: Use a more advanced distortion model for high-res cameras ---
    # cv.CALIB_RATIONAL_MODEL enables an 8-coefficient model (k1-k6, p1, p2),
    # which is more suitable for complex lens distortions.
    print("\nCalibrating with 5-parameter rational distortion model...")
    # calibration_flags = cv.CALIB_RATIONAL_MODEL
    # ret, cmtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None, flags=calibration_flags)
    # 修改后：使用默认的5参数模型（删除flags参数）
    ret, cmtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)  # 不指定flags，默认使用5参数模型
    print('\n--- Calibration Results ---')
    print('Overall RMSE:', ret)
    print('Camera matrix (K):\n', cmtx)
    print('Distortion coeffs (k1,k2,p1,p2,k3):\n', dist.ravel())

    # --- MODIFICATION 2: Add per-view error analysis for diagnostics ---
    # This is crucial to identify and remove problematic images that hurt calibration quality.
    print("\n--- Per-View Reprojection Errors ---")
    print("Check for images with significantly higher error than the average.")
    mean_error = 0 
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cmtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        print(f"Image '{used_image_names[i]}': {error:.4f} pixels")
        mean_error += error

    print(f"\nAverage error calculated manually: {mean_error/len(objpoints):.4f} pixels")
    print("---------------------------------\n")

    return cmtx, dist


def stereo_calibrate(mtx0, dist0, mtx1, dist1, image_paths_c0, image_paths_c1, screen_res):
    """Stereo calibrate... (Modified for better display)"""
    c0_images = [cv.imread(imname, 1) for imname in image_paths_c0]
    c1_images = [cv.imread(imname, 1) for imname in image_paths_c1]

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    rows, columns = calibration_settings['checkerboard_rows'], calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale']

    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling * objp

    width, height = c0_images[0].shape[1], c0_images[0].shape[0]
    imgpoints_left, imgpoints_right, objpoints = [], [], []

    display_size_single = calculate_optimal_display_size(width, height, screen_res[0], screen_res[1], num_images_h=2)
    window_name = 'Stereo Pair - Press "s" to skip, any other key to continue'
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, display_size_single[0] * 2, display_size_single[1])

    print(f"Processing {len(c0_images)} image pairs for stereo calibration...")
    for frame0, frame1 in zip(c0_images, c1_images):
        gray1 = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (rows, columns), None)

        if c_ret1 and c_ret2:
            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
            cv.drawChessboardCorners(frame0, (rows,columns), corners1, c_ret1)
            cv.drawChessboardCorners(frame1, (rows,columns), corners2, c_ret2)
            # 在cv.drawChessboardCorners之后，添加角点序号标注
            for j, corner in enumerate(corners1):
                cv.putText(frame0, f"{j}", (int(corner[0][0]), int(corner[0][1])), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            for j, corner in enumerate(corners2):
                cv.putText(frame1, f"{j}", (int(corner[0][0]), int(corner[0][1])), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            display_frame0 = cv.resize(frame0, display_size_single)
            display_frame1 = cv.resize(frame1, display_size_single)
            cv.imshow(window_name, np.hstack([display_frame0, display_frame1]))
            
            k = cv.waitKey(0)
            if k & 0xFF == ord('s'):
                print('Skipping this pair'); continue

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    cv.destroyAllWindows()
    if not objpoints:
        print("Error: No valid checkerboard pairs found. Cannot perform stereo calibration."); quit()
        
    flags = cv.CALIB_FIX_INTRINSIC
    # The rational model is already part of the intrinsic distortion coefficients (dist0, dist1)
    # So we don't need to add the flag again here.
    ret, _, _, _, _, R, T, _, _ = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx0, dist0,
                                                      mtx1, dist1, (width, height), criteria=criteria, flags=flags)
    print('\nStereo calibration RMSE: ', ret)
    return R, T

def save_camera_intrinsics(camera_matrix, distortion_coefs, camera_name):
    if not os.path.exists('camera_parameters'): os.mkdir('camera_parameters')
    out_filename = os.path.join('camera_parameters', camera_name + '_intrinsics.dat')
    with open(out_filename, 'w') as outf:
        outf.write('intrinsic:\n'); np.savetxt(outf, camera_matrix, fmt='%f')
        outf.write('distortion:\n'); np.savetxt(outf, distortion_coefs, fmt='%f')
    print(f"Saved intrinsics for {camera_name} to {out_filename}")

def save_extrinsic_calibration_parameters(R, T, camera0_name='camera0', camera1_name='camera1'):
    if not os.path.exists('camera_parameters'): os.mkdir('camera_parameters')
    filename = os.path.join('camera_parameters', camera1_name + '_rot_trans.dat')
    with open(filename, 'w') as f:
        f.write('R:\n'); np.savetxt(f, R, fmt='%f')
        f.write('T:\n'); np.savetxt(f, T, fmt='%f')
    print(f"Saved extrinsics (R, T) to {filename}")

def find_and_pair_images(base_dir, ext):
    left_subdir = os.path.join(base_dir, 'left')
    right_subdir = os.path.join(base_dir, 'right')
    if not os.path.isdir(left_subdir) or not os.path.isdir(right_subdir):
        print(f"Error: Subdirectories 'left' and/or 'right' not found inside '{base_dir}'"); quit()
    all_left_paths = sorted(glob.glob(os.path.join(left_subdir, f'*.{ext}')))
    all_right_paths = sorted(glob.glob(os.path.join(right_subdir, f'*.{ext}')))
    if not all_left_paths or not all_right_paths:
        print(f"Error: No images with extension '.{ext}' found in subdirectories."); quit()
    print(f"Found {len(all_left_paths)} images in camera_0 and {len(all_right_paths)} images in camera_1.")
    left_image_map = {}
    for path in all_left_paths:
        try:
            identifier = os.path.basename(path).split('_')[-1].split('.')[0]
            left_image_map[identifier] = path
        except IndexError:
            print(f"Warning: Could not parse identifier from filename: {os.path.basename(path)}. Skipping."); continue
    paired_left_paths, paired_right_paths = [], []
    for right_path in all_right_paths:
        try:
            identifier = os.path.basename(right_path).split('_')[-1].split('.')[0]
            if identifier in left_image_map:
                paired_left_paths.append(left_image_map[identifier])
                paired_right_paths.append(right_path)
        except IndexError: continue
    print(f"Successfully matched {len(paired_left_paths)} image pairs.")
    if not paired_left_paths:
        print("Error: No matching image pairs found. Check your filenames."); quit()
    return paired_left_paths, paired_right_paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calibrate a stereo camera system from image folders.")
    parser.add_argument('settings_file', type=str, help='Path to the calibration_settings.yaml file.')
    parser.add_argument('base_dir', type=str, help="Path to the base directory containing 'camera_0' and 'camera_1' subfolders.")
    parser.add_argument('--ext', type=str, default='bmp', help='Extension of the image files (e.g., bmp, png, jpg).')
    args = parser.parse_args()

    parse_calibration_settings_file(args.settings_file)

    screen_resolution = get_screen_resolution()
    print(f"Detected screen resolution: {screen_resolution[0]}x{screen_resolution[1]}")

    left_image_paths, right_image_paths = find_and_pair_images(args.base_dir, args.ext)

    print("\n--- Calibrating Intrinsics for Left Camera (camera_0) ---")
    cmtx0, dist0 = calibrate_camera_for_intrinsic_parameters(left_image_paths, screen_resolution)
    save_camera_intrinsics(cmtx0, dist0, 'camera0')

    print("\n--- Calibrating Intrinsics for Right Camera (camera_1) ---")
    cmtx1, dist1 = calibrate_camera_for_intrinsic_parameters(right_image_paths, screen_resolution)
    save_camera_intrinsics(cmtx1, dist1, 'camera1')

    print("\n--- Performing Stereo Calibration ---")
    R, T = stereo_calibrate(cmtx0, dist0, cmtx1, dist1, left_image_paths, right_image_paths, screen_resolution)

    print("\n--- Saving Extrinsic Parameters ---")
    save_extrinsic_calibration_parameters(R, T)
    
    print("\nCalibration complete. Parameters saved to 'camera_parameters' directory.")