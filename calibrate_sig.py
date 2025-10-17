
import cv2 as cv
import glob
import numpy as np
import sys
from scipy import linalg
import yaml
import os

#This will contain the calibration settings from the calibration_settings.yaml file
calibration_settings = {}

#Given Projection matrices P1 and P2, and pixel coordinates point1 and point2, return triangulated 3D point.
def DLT(P1, P2, point1, point2):

    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))

    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices = False)

    #print('Triangulated point: ')
    #print(Vh[3,0:3]/Vh[3,3])
    return Vh[3,0:3]/Vh[3,3]


#Open and load the calibration_settings.yaml file
def parse_calibration_settings_file(filename):
    
    global calibration_settings

    if not os.path.exists(filename):
        print('File does not exist:', filename)
        quit()
    
    print('Using for calibration settings: ', filename)

    with open(filename) as f:
        calibration_settings = yaml.safe_load(f)

    #rudimentray check to make sure correct file was loaded
    if 'stereo_camera_device_id' not in calibration_settings.keys():
        print('stereo_camera_device_id key was not found in the settings file.')
        print('Please add it for the split-frame stereo camera. Check documentation.')
        quit()


# <<<< NEW FUNCTION TO HANDLE SPLIT-FRAME STEREO CAMERA >>>>
# This function replaces both save_frames_single_camera and save_frames_two_cams
def save_frames_from_split_stereo_camera(camera0_name, camera1_name, save_dir, num_frames_setting_key):
    """
    Opens a single stereo camera that produces a wide frame, splits it into
    left (camera0) and right (camera1) views, and saves calibration frames.
    Saves files into a subdirectory relative to the current working directory.
    """
    # --- MODIFICATION: Ensure save_dir is relative to the current working directory ---
    # Get the current working directory
    current_working_dir = os.getcwd()
    # Create the full path for the save directory
    full_save_dir = os.path.join(current_working_dir, save_dir)

    # Create save directory if it doesn't exist
    if not os.path.exists(full_save_dir):
        print(f"Creating directory: {full_save_dir}")
        os.makedirs(full_save_dir)

    # Get settings from the YAML file
    camera_device_id = calibration_settings['stereo_camera_device_id']
    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    number_to_save = calibration_settings[num_frames_setting_key]
    view_resize = calibration_settings['view_resize']
    cooldown_time = calibration_settings['cooldown']

    # Open the single video stream
    cap = cv.VideoCapture(camera_device_id, cv.CAP_DSHOW) # Use CAP_DSHOW for better compatibility on Windows
    if not cap.isOpened():
        print(f"Error: Could not open camera with ID {camera_device_id}")
        quit()
        
    # Set the desired wide resolution
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    
    # Verify the resolution
    actual_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    print(f"Requested resolution: {width}x{height}")
    print(f"Actual resolution: {actual_width}x{actual_height}")
    if actual_width != width or actual_height != height:
        print("Warning: Camera did not accept the requested resolution.")

    cooldown = cooldown_time
    start = False
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No video data received from camera. Exiting...")
            break

        # --- Core modification: Split the wide frame into two halves ---
        if frame.shape[1] != width:
             print(f"Warning: Frame width ({frame.shape[1]}) does not match expected width ({width}). Skipping split.")
             cv.imshow('Full Frame (Error)', frame)
        else:
            half_width = frame.shape[1] // 2
            frame0 = frame[:, :half_width] # Left half
            frame1 = frame[:, half_width:] # Right half

            frame0_small = cv.resize(frame0, None, fx=1/view_resize, fy=1/view_resize)
            frame1_small = cv.resize(frame1, None, fx=1/view_resize, fy=1/view_resize)

            if not start:
                cv.putText(frame0_small, "Press SPACEBAR to start collection", (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            
            if start:
                cooldown -= 1
                cv.putText(frame0_small, f"Cooldown: {cooldown}", (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv.putText(frame0_small, f"Saved: {saved_count}/{number_to_save}", (50, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv.putText(frame1_small, f"Saved: {saved_count}/{number_to_save}", (50, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                
                if cooldown <= 0:
                    # Save both left and right frames using the full path
                    savename0 = os.path.join(full_save_dir, f"{camera0_name}_{saved_count}.png")
                    cv.imwrite(savename0, frame0)
                    savename1 = os.path.join(full_save_dir, f"{camera1_name}_{saved_count}.png")
                    cv.imwrite(savename1, frame1)
                    
                    print(f"Saved {savename0} and {savename1}")
                    
                    saved_count += 1
                    cooldown = cooldown_time

            cv.imshow('Left Eye (camera0)', frame0_small)
            cv.imshow('Right Eye (camera1)', frame1_small)

        k = cv.waitKey(1)
        
        if k == 27: # ESC key
            print("ESC pressed. Exiting.")
            break
        if k == 32: # Spacebar
            print("Starting frame collection...")
            start = True

        if saved_count >= number_to_save:
            print(f"Successfully saved {saved_count} frame pairs.")
            break

    cap.release()
    cv.destroyAllWindows()


# --- NO CHANGES ARE NEEDED FOR THE FUNCTIONS BELOW THIS LINE ---

#Calibrate single camera to obtain camera intrinsic parameters from saved frames.
def calibrate_camera_for_intrinsic_parameters(images_prefix):
    
    # NOTE: images_prefix will now be a relative path like 'frames/camera0*', which is correct.
    images_names = glob.glob(images_prefix)
    if not images_names:
        print(f"Error: No images found with prefix '{images_prefix}'. Make sure images were saved correctly.")
        quit()

    #read all frames
    images = [cv.imread(imname, 1) for imname in images_names]

    #criteria used by checkerboard pattern detector.
    #Change this if the code can't find the checkerboard. 
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale'] #this will change to user defined length scale

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    #frame dimensions. Frames should be the same size.
    width = images[0].shape[1]
    height = images[0].shape[0]

    #Pixel coordinates of checkerboards
    imgpoints = [] # 2d points in image plane.

    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space

    print(f"Found {len(images)} images for calibration.")
    for i, frame in enumerate(images):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        #find the checkerboard
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

        if ret == True:
            #Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)

            #opencv can attempt to improve the checkerboard coordinates
            corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            cv.drawChessboardCorners(frame, (rows,columns), corners, ret)
            cv.putText(frame, 'If detected points are poor, press "s" to skip this sample', (25, 25), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)

            cv.imshow('img', frame)
            k = cv.waitKey(0)

            if k & 0xFF == ord('s'):
                print(f'Skipping image {i}')
                continue

            objpoints.append(objp)
            imgpoints.append(corners)

    cv.destroyAllWindows()
    
    if not objpoints or not imgpoints:
        print("Error: No valid checkerboard points found. Cannot calibrate.")
        quit()
        
    ret, cmtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    print('rmse:', ret)
    print('camera matrix:\n', cmtx)
    print('distortion coeffs:', dist)

    return cmtx, dist

#save camera intrinsic parameters to file
def save_camera_intrinsics(camera_matrix, distortion_coefs, camera_name, save_dir='camera_parameters'):
    
    # --- MODIFICATION: Ensure save_dir is relative to the current working directory ---
    if not os.path.exists(save_dir):
        print(f"Creating directory: {save_dir}")
        os.makedirs(save_dir)

    out_filename = os.path.join(save_dir, camera_name + '_intrinsics.dat')
    with open(out_filename, 'w') as outf:
        outf.write('intrinsic:\n')
        for l in camera_matrix:
            for en in l:
                outf.write(str(en) + ' ')
            outf.write('\n')

        outf.write('distortion:\n')
        for en in distortion_coefs[0]:
            outf.write(str(en) + ' ')
        outf.write('\n')
    print(f"Saved camera intrinsics to {out_filename}")


#open paired calibration frames and stereo calibrate for cam0 to cam1 coorindate transformations
def stereo_calibrate(mtx0, dist0, mtx1, dist1, frames_prefix_c0, frames_prefix_c1):
    #read the synched frames
    c0_images_names = sorted(glob.glob(frames_prefix_c0))
    c1_images_names = sorted(glob.glob(frames_prefix_c1))

    if not c0_images_names or not c1_images_names:
        print(f"Error: No images found for stereo calibration. Prefixes: '{frames_prefix_c0}', '{frames_prefix_c1}'")
        quit()

    #open images
    c0_images = [cv.imread(imname, 1) for imname in c0_images_names]
    c1_images = [cv.imread(imname, 1) for imname in c1_images_names]

    #change this if stereo calibration not good.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    #calibration pattern settings
    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale']

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    #frame dimensions. Frames should be the same size.
    width = c0_images[0].shape[1]
    height = c0_images[0].shape[0]

    #Pixel coordinates of checkerboards
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []

    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space

    for i, (frame0, frame1) in enumerate(zip(c0_images, c1_images)):
        gray1 = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (rows, columns), None)

        if c_ret1 == True and c_ret2 == True:

            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            p0_c1 = corners1[0,0].astype(np.int32)
            p0_c2 = corners2[0,0].astype(np.int32)

            cv.putText(frame0, 'O', (p0_c1[0], p0_c1[1]), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
            cv.drawChessboardCorners(frame0, (rows,columns), corners1, c_ret1)
            cv.imshow('img', frame0)

            cv.putText(frame1, 'O', (p0_c2[0], p0_c2[1]), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
            cv.drawChessboardCorners(frame1, (rows,columns), corners2, c_ret2)
            cv.imshow('img2', frame1)
            k = cv.waitKey(0)

            if k & 0xFF == ord('s'):
                print(f'Skipping image pair {i}')
                continue

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    cv.destroyAllWindows()
    
    if not objpoints:
        print("Error: No valid checkerboard point pairs found for stereo calibration.")
        quit()
        
    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist0, CM2, dist1, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx0, dist0,
                                                                 mtx1, dist1, (width, height), criteria = criteria, flags = stereocalibration_flags)

    print('rmse: ', ret)
    return R, T

#Converts Rotation matrix R and Translation vector T into a homogeneous representation matrix
def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4,4))
    P[:3,:3] = R
    P[:3, 3] = t.reshape(3)
    P[3,3] = 1
 
    return P
# Turn camera calibration data into projection matrix
def get_projection_matrix(cmtx, R, T):
    P = cmtx @ _make_homogeneous_rep_matrix(R, T)[:3,:]
    return P

# After calibrating, we can see shifted coordinate axes in the video feeds directly
def check_calibration(camera0_name, camera0_data, camera1_name, camera1_data, _zshift = 50.):
    
    cmtx0 = np.array(camera0_data[0])
    dist0 = np.array(camera0_data[1])
    R0 = np.array(camera0_data[2])
    T0 = np.array(camera0_data[3])
    cmtx1 = np.array(camera1_data[0])
    dist1 = np.array(camera1_data[1])
    R1 = np.array(camera1_data[2])
    T1 = np.array(camera1_data[3])

    P0 = get_projection_matrix(cmtx0, R0, T0)
    P1 = get_projection_matrix(cmtx1, R1, T1)

    #define coordinate axes in 3D space. These are just the usual coorindate vectors
    coordinate_points = np.array([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
    z_shift = np.array([0.,0.,_zshift]).reshape((1, 3))
    #increase the size of the coorindate axes and shift in the z direction
    draw_axes_points = 5 * coordinate_points + z_shift

    #project 3D points to each camera view manually. This can also be done using cv.projectPoints()
    #Note that this uses homogenous coordinate formulation
    pixel_points_camera0 = []
    pixel_points_camera1 = []
    for _p in draw_axes_points:
        X = np.array([_p[0], _p[1], _p[2], 1.])
        
        uv = P0 @ X
        uv = np.array([uv[0], uv[1]])/uv[2]
        pixel_points_camera0.append(uv)

        uv = P1 @ X
        uv = np.array([uv[0], uv[1]])/uv[2]
        pixel_points_camera1.append(uv)

    pixel_points_camera0 = np.array(pixel_points_camera0)
    pixel_points_camera1 = np.array(pixel_points_camera1)

    cap = cv.VideoCapture(calibration_settings['stereo_camera_device_id'], cv.CAP_DSHOW)
    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Video stream not returning frame data')
            quit()

        half_width = frame.shape[1] // 2
        frame0 = frame[:, :half_width]
        frame1 = frame[:, half_width:]

        colors = [(0,0,255), (0,255,0), (255,0,0)] # BGR for Red, Green, Blue
        # Draw projections to camera0 (X=Red, Y=Green, Z=Blue)
        origin = tuple(pixel_points_camera0[0].astype(np.int32))
        cv.line(frame0, origin, tuple(pixel_points_camera0[1].astype(np.int32)), colors[0], 2) # X
        cv.line(frame0, origin, tuple(pixel_points_camera0[2].astype(np.int32)), colors[1], 2) # Y
        cv.line(frame0, origin, tuple(pixel_points_camera0[3].astype(np.int32)), colors[2], 2) # Z
        
        # Draw projections to camera1
        origin = tuple(pixel_points_camera1[0].astype(np.int32))
        cv.line(frame1, origin, tuple(pixel_points_camera1[1].astype(np.int32)), colors[0], 2) # X
        cv.line(frame1, origin, tuple(pixel_points_camera1[2].astype(np.int32)), colors[1], 2) # Y
        cv.line(frame1, origin, tuple(pixel_points_camera1[3].astype(np.int32)), colors[2], 2) # Z

        cv.imshow('frame0', frame0)
        cv.imshow('frame1', frame1)

        k = cv.waitKey(1)
        if k == 27: break
    
    cap.release()
    cv.destroyAllWindows()


def save_extrinsic_calibration_parameters(R0, T0, R1, T1, save_dir='camera_parameters', prefix = ''):
    
    # --- MODIFICATION: Ensure save_dir is relative to the current working directory ---
    if not os.path.exists(save_dir):
        print(f"Creating directory: {save_dir}")
        os.makedirs(save_dir)

    camera0_rot_trans_filename = os.path.join(save_dir, prefix + 'camera0_rot_trans.dat')
    with open(camera0_rot_trans_filename, 'w') as outf:
        outf.write('R:\n')
        for l in R0:
            for en in l:
                outf.write(str(en) + ' ')
            outf.write('\n')

        outf.write('T:\n')
        for l in T0:
            for en in l:
                outf.write(str(en) + ' ')
            outf.write('\n')
    print(f"Saved camera0 extrinsics to {camera0_rot_trans_filename}")

    camera1_rot_trans_filename = os.path.join(save_dir, prefix + 'camera1_rot_trans.dat')
    with open(camera1_rot_trans_filename, 'w') as outf:
        outf.write('R:\n')
        for l in R1:
            for en in l:
                outf.write(str(en) + ' ')
            outf.write('\n')

        outf.write('T:\n')
        for l in T1:
            for en in l:
                outf.write(str(en) + ' ')
            outf.write('\n')
    print(f"Saved camera1 extrinsics to {camera1_rot_trans_filename}")


if __name__ == '__main__':
    
    settings_filename = 'calibration_settings_sigle.yaml'
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Build the absolute path to the settings file
    settings_filepath = os.path.join(script_dir, settings_filename)

    print(f"Attempting to load settings from: {settings_filepath}")
    # Open and parse the settings file
    parse_calibration_settings_file(settings_filepath)
    
    # Define relative paths for output directories
    intrinsic_images_dir = 'frames'
    stereo_images_dir = 'frames_pair'
    params_dir = 'camera_parameters'

    """Step1. Save calibration frames for intrinsic calibration."""
    print("\n--- Step 1: Capturing frames for INTRINSIC calibration ---")
    save_frames_from_split_stereo_camera('camera0', 'camera1', intrinsic_images_dir, 'mono_calibration_frames')

    """Step2. Obtain camera intrinsic matrices and save them."""
    print("\n--- Step 2: Calculating INTRINSIC parameters for each camera ---")
    # Camera0 intrinsics
    print("\nCalibrating camera0...")
    images_prefix_c0 = os.path.join(intrinsic_images_dir, 'camera0*')
    cmtx0, dist0 = calibrate_camera_for_intrinsic_parameters(images_prefix_c0) 
    save_camera_intrinsics(cmtx0, dist0, 'camera0', save_dir=params_dir)
    
    # Camera1 intrinsics
    print("\nCalibrating camera1...")
    images_prefix_c1 = os.path.join(intrinsic_images_dir, 'camera1*')
    cmtx1, dist1 = calibrate_camera_for_intrinsic_parameters(images_prefix_c1)
    save_camera_intrinsics(cmtx1, dist1, 'camera1', save_dir=params_dir)

    """Step3. Save paired calibration frames for stereo calibration."""
    print("\n--- Step 3: Capturing frames for STEREO calibration ---")
    save_frames_from_split_stereo_camera('camera0', 'camera1', stereo_images_dir, 'stereo_calibration_frames')

    """Step4. Use paired frames to obtain camera0 to camera1 transformation."""
    print("\n--- Step 4: Calculating STEREO parameters (extrinsics) ---")
    frames_prefix_c0 = os.path.join(stereo_images_dir, 'camera0*')
    frames_prefix_c1 = os.path.join(stereo_images_dir, 'camera1*')
    R, T = stereo_calibrate(cmtx0, dist0, cmtx1, dist1, frames_prefix_c0, frames_prefix_c1)

    """Step5. Save calibration data and verify."""
    print("\n--- Step 5: Saving and Verifying Calibration ---")
    # Define world coordinate system at camera0
    R0 = np.eye(3, dtype=np.float32)
    T0 = np.array([0., 0., 0.]).reshape((3, 1))
    R1 = R
    T1 = T
    
    save_extrinsic_calibration_parameters(R0, T0, R1, T1, save_dir=params_dir)
    
    # Check calibration visually
    camera0_data = [cmtx0, dist0, R0, T0]
    camera1_data = [cmtx1, dist1, R1, T1]
    print("Calibration saved. Press ESC in the video window to exit the verification.")
    check_calibration('camera0', camera0_data, 'camera1', camera1_data, _zshift = 60.)

    print("\n--- Calibration process complete! ---")