# --- START OF FILE MultipleCameras.py ---

# -*- coding: utf-8 -*-
import sys
import time
import os
import threading
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QTextCursor, QIcon
from CamOperation_class import CameraOperation
from MvCameraControl_class import *
from MvErrorDefine_const import *
from CameraParams_header import *
from PyUIMultipleCameras import Ui_MainWindow
import ctypes

# Define the number of cameras the UI will manage.
NUM_CAMERAS = 2

def ToHexStr(num):
    chaDic = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
    hexStr = ""
    if num < 0:
        num = num + 2 ** 32
    while num >= 16:
        digit = num % 16
        hexStr = chaDic.get(digit, str(digit)) + hexStr
        num //= 16
    hexStr = chaDic.get(num, str(num)) + hexStr
    return hexStr

def decoding_char(c_ubyte_value):
    c_char_p_value = ctypes.cast(c_ubyte_value, ctypes.c_char_p)
    try:
        decode_str = c_char_p_value.value.decode('gbk')
    except UnicodeDecodeError:
        decode_str = str(c_char_p_value.value)
    return decode_str


if __name__ == "__main__":

    global deviceList
    deviceList = MV_CC_DEVICE_INFO_LIST()
    global cam_checked_list
    cam_checked_list = []
    global obj_cam_operation
    obj_cam_operation = []
    global win_display_handles
    win_display_handles = []
    global valid_number
    valid_number = 0
    global b_is_open
    b_is_open = False
    global b_is_grab
    b_is_grab = False
    global b_is_trigger
    b_is_trigger = False
    global b_is_software_trigger
    b_is_software_trigger = False

    MvCamera.MV_CC_Initialize()

    def print_text(str_info):
        ui.textEdit.moveCursor(QTextCursor.Start)
        ui.textEdit.insertPlainText(str_info + "\n")

    def enum_devices():
        global deviceList, valid_number
        deviceList = MV_CC_DEVICE_INFO_LIST()
        n_layer_type = (MV_GIGE_DEVICE | MV_USB_DEVICE
                        | MV_GENTL_GIGE_DEVICE | MV_GENTL_CAMERALINK_DEVICE
                        | MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE)
        ret = MvCamera.MV_CC_EnumDevicesEx2(n_layer_type, deviceList, '', SortMethod_SerialNumber)
        if ret != 0:
            str_error = "Enum devices fail! ret = :" + ToHexStr(ret)
            QMessageBox.warning(mainWindow, "Error", str_error, QMessageBox.Ok)
            return ret

        if deviceList.nDeviceNum == 0:
            QMessageBox.warning(mainWindow, "Info", "Find no device", QMessageBox.Ok)
            return ret
        print_text("Find %d devices!" % deviceList.nDeviceNum)

        valid_number = 0
        for i in range(0, NUM_CAMERAS):
            if (i < deviceList.nDeviceNum) is True:
                serial_number = ""
                model_name = ""
                mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
                if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE or mvcc_dev_info.nTLayerType == MV_GENTL_GIGE_DEVICE:
                    model_name = decoding_char(mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName)
                    for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chSerialNumber:
                        if per == 0: break
                        serial_number = serial_number + chr(per)
                elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                    model_name = decoding_char(mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName)
                    for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                        if per == 0: break
                        serial_number = serial_number + chr(per)
                
                button_by_id = cam_button_group.button(i)
                button_by_id.setText(f"Cam{i+1}: {model_name} ({serial_number})")
                button_by_id.setEnabled(True)
                valid_number = valid_number + 1
            else:
                button_by_id = cam_button_group.button(i)
                button_by_id.setText(f"Camera {i+1}")
                button_by_id.setEnabled(False)

    def cam_check_box_clicked():
        global cam_checked_list
        cam_checked_list = []
        for i in range(0, NUM_CAMERAS):
            button = cam_button_group.button(i)
            if button.isChecked() is True:
                cam_checked_list.append(True)
            else:
                cam_checked_list.append(False)

    def enable_ui_controls():
        global b_is_open, b_is_grab, b_is_trigger, b_is_software_trigger
        
        # --- MODIFICATION START ---
        # The logic for enabling/disabling UI elements has been corrected and clarified.

        # Buttons are enabled only when the device is NOT open.
        ui.pushButton_enum.setEnabled(not b_is_open)
        ui.pushButton_open.setEnabled(not b_is_open)
        
        # The close button is enabled only when the device IS open.
        ui.pushButton_close.setEnabled(b_is_open)
        
        # This state represents when the camera is ready to grab, but not currently grabbing.
        # This is the ONLY state where changing parameters should be allowed.
        is_configurable = b_is_open and not b_is_grab
        
        # This state represents when the camera is currently grabbing.
        is_grabbing = b_is_open and b_is_grab
        
        # Start button is enabled only when ready to grab.
        ui.pushButton_startGrab.setEnabled(is_configurable)
        
        # Stop and Save buttons are enabled only when actively grabbing.
        ui.pushButton_stopGrab.setEnabled(is_grabbing)
        ui.pushButton_saveImg.setEnabled(is_grabbing)
        
        # Parameter and trigger mode settings are enabled ONLY when configurable (open but not grabbing).
        # This is the core fix for the bug.
        ui.radioButton_continuous.setEnabled(is_configurable)
        ui.radioButton_trigger.setEnabled(is_configurable)
        ui.group_params.setEnabled(is_configurable)
        
        # The trigger source dropdown is only active if we are in trigger mode AND configurable.
        is_trigger_mode_active = b_is_trigger and is_configurable
        ui.label_triggerSource.setEnabled(is_trigger_mode_active)
        ui.comboBox_triggerSource.setEnabled(is_trigger_mode_active)
        
        # The software trigger button is active only if all conditions are met:
        # device is grabbing, it's in trigger mode, and the source is software.
        ui.pushButton_triggerOnce.setEnabled(is_grabbing and b_is_trigger and b_is_software_trigger)
        
        # --- MODIFICATION END ---


    def open_devices():
        global deviceList, obj_cam_operation, b_is_open, valid_number, cam_checked_list
        cam_check_box_clicked()
        if b_is_open is True:
            return
        if len(cam_checked_list) <= 0 or not any(cam_checked_list):
            print_text("Please select at least one camera to open!")
            return
            
        obj_cam_operation = []
        opened_count = 0
        for i in range(0, NUM_CAMERAS):
            if i < len(cam_checked_list) and cam_checked_list[i] is True:
                op = CameraOperation(None, deviceList, i)
                ret = op.open_device()
                if 0 != ret:
                    print_text(f"Open camera {i+1} failed! ret[0x{ToHexStr(ret)}]")
                    obj_cam_operation.append(0)
                else:
                    print_text(f"Open camera {i+1} successfully.")
                    obj_cam_operation.append(op)
                    opened_count += 1
            else:
                obj_cam_operation.append(0)

        if opened_count > 0:
            b_is_open = True
            ui.radioButton_continuous.setChecked(True)
            radio_button_clicked(None) 
            for i in range(0, valid_number):
                cam_button_group.button(i).setEnabled(False)
        else:
            print_text("No cameras were opened successfully.")
            b_is_open = False
        enable_ui_controls()

    def trigger_source_changed(source_text):
        global obj_cam_operation, b_is_software_trigger
        api_string = source_text.lower()
        b_is_software_trigger = (api_string == "software")

        for i in range(0, NUM_CAMERAS):
            if obj_cam_operation[i] != 0:
                ret = obj_cam_operation[i].set_trigger_source(api_string)
                if 0 != ret:
                    print_text(f'Cam {i+1} set trigger source to {source_text} failed! ret={ToHexStr(ret)}')
        enable_ui_controls()

    def radio_button_clicked(button):
        global obj_cam_operation, b_is_trigger
        b_is_trigger = ui.radioButton_trigger.isChecked()
        mode_str = "triggermode" if b_is_trigger else "continuous"
        
        for i in range(0, NUM_CAMERAS):
            if obj_cam_operation[i] != 0:
                ret = obj_cam_operation[i].set_trigger_mode(mode_str)
                if 0 != ret:
                    print_text(f'Cam {i+1} set mode to {mode_str} failed! ret={ToHexStr(ret)}')
        
        if b_is_trigger:
            trigger_source_changed(ui.comboBox_triggerSource.currentText())
        
        enable_ui_controls()

    def close_devices():
        global b_is_open, obj_cam_operation, valid_number
        if not b_is_open:
            return
        if b_is_grab:
            print_text(f'stop grab before close device')
            stop_grabbing()
        for i in range(0, NUM_CAMERAS):
            if obj_cam_operation[i] != 0:
                obj_cam_operation[i].close_device()
        for i in range(0, valid_number):
            cam_button_group.button(i).setEnabled(True)
        b_is_open = False
        enable_ui_controls()

    def start_grabbing():
        global obj_cam_operation, win_display_handles, b_is_open, b_is_grab
        if (not b_is_open) or (b_is_grab):
            return
        ui.pushButton_startGrab.setEnabled(False)
        ui.pushButton_stopGrab.setEnabled(False)
        QApplication.processEvents()  # 强制UI刷新，确保用户看到按钮被禁用
        grab_ok = False
        for i in range(0, NUM_CAMERAS):
            if obj_cam_operation[i] != 0:
                ret = obj_cam_operation[i].start_grabbing(i, win_display_handles[i])
                if 0 != ret:
                    print_text(f'Camera {i+1} start grabbing failed! ret={ToHexStr(ret)}')
                else:
                    grab_ok = True
        if grab_ok:
            b_is_grab = True
        enable_ui_controls()

    def stop_grabbing():
        global b_is_grab, obj_cam_operation, b_is_open
        if (not b_is_open) or (not b_is_grab):
            return
        for i in range(0, NUM_CAMERAS):
            if obj_cam_operation[i] != 0:
                obj_cam_operation[i].stop_grabbing()
        b_is_grab = False
        enable_ui_controls()

    def save_bmp():
        global b_is_grab, obj_cam_operation
        if not b_is_grab:
            return
        # 在停止操作前，立即禁用按钮
        ui.pushButton_startGrab.setEnabled(False)
        ui.pushButton_stopGrab.setEnabled(False)
        QApplication.processEvents()  # 强制UI刷新
        base_save_folder = os.path.join(os.getcwd(), "saved_images")
        os.makedirs(base_save_folder, exist_ok=True)
        for i in range(0, NUM_CAMERAS):
            if obj_cam_operation[i] != 0:
                ret = obj_cam_operation[i].save_bmp(base_save_folder)
                if 0 != ret:
                    print_text(f'Camera {i+1} save bmp failed! ret={ToHexStr(ret)}')
                else:
                    print_text(f'Camera {i+1} image saved.')

    def is_float(str_value):
        try:
            float(str_value)
            return True
        except ValueError:
            return False
    def set_parameters():
        global obj_cam_operation, b_is_open
        if not b_is_open:
            return

        exposure_time_str = ui.lineEdit_exposureTime.text()
        gain_str = ui.lineEdit_gain.text()
        frame_rate_str = ui.lineEdit_frameRate.text()

        # 用于记录哪些参数被成功应用
        params_successfully_set = []

        # 1. 独立检查并设置曝光时间
        if exposure_time_str:  # 首先检查输入框是否为空
            if is_float(exposure_time_str):
                for i in range(0, NUM_CAMERAS):
                    if obj_cam_operation[i] != 0:
                        obj_cam_operation[i].set_exposure_time(exposure_time_str)
                params_successfully_set.append("Exposure")
            else:
                print_text(f"错误: 曝光时间值 '{exposure_time_str}' 无效，未应用。")

        # 2. 独立检查并设置增益
        if gain_str:  # 检查增益输入框是否为空
            if is_float(gain_str):
                for i in range(0, NUM_CAMERAS):
                    if obj_cam_operation[i] != 0:
                        obj_cam_operation[i].set_gain(gain_str)
                params_successfully_set.append("Gain")
            else:
                print_text(f"错误: 增益值 '{gain_str}' 无效，未应用。")

        # 3. 独立检查并设置帧率
        if frame_rate_str:  # 检查帧率输入框是否为空
            if is_float(frame_rate_str):
                for i in range(0, NUM_CAMERAS):
                    if obj_cam_operation[i] != 0:
                        obj_cam_operation[i].set_frame_rate(frame_rate_str)
                params_successfully_set.append("Frame Rate")
            else:
                print_text(f"错误: 帧率值 '{frame_rate_str}' 无效，未应用。")

        # 根据是否有参数被成功设置来提供反馈
        if params_successfully_set:
            # 使用 ', '.join() 来创建一个清晰的列表，例如 "Exposure, Gain"
            print_text(f"成功应用参数: {', '.join(params_successfully_set)}.")
        else:
            # 如果所有输入框都为空，或者填写的值都无效，则给出此提示
            print_text("没有输入任何有效的参数。")
    def trigger_camera_thread(cam_op, cam_index):
        ret = cam_op.trigger_once()
        if ret != 0:
            print(f'Camera {cam_index+1} software trigger failed: ret {ToHexStr(ret)}')

    def software_trigger_once():
        threads = []
        for i in range(0, NUM_CAMERAS):
            if obj_cam_operation[i] != 0:
                thread = threading.Thread(target=trigger_camera_thread, args=(obj_cam_operation[i], i))
                threads.append(thread)
        for thread in threads:
            thread.start()
            
        enable_ui_controls()
        
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(mainWindow)
    try:
        ui.pushButton_enum.clicked.connect(enum_devices)
        ui.pushButton_open.clicked.connect(open_devices)
        ui.pushButton_close.clicked.connect(close_devices)
        ui.pushButton_startGrab.clicked.connect(start_grabbing)
        ui.pushButton_stopGrab.clicked.connect(stop_grabbing)
        ui.pushButton_saveImg.clicked.connect(save_bmp)
        ui.pushButton_setParams.clicked.connect(set_parameters)
        ui.pushButton_triggerOnce.clicked.connect(software_trigger_once)
        
        ui.comboBox_triggerSource.currentTextChanged.connect(trigger_source_changed)
        
        cam_button_group = QButtonGroup(mainWindow)
        cam_button_group.addButton(ui.checkBox_1, 0)
        cam_button_group.addButton(ui.checkBox_2, 1)
        cam_button_group.setExclusive(False)
        cam_button_group.buttonClicked.connect(cam_check_box_clicked)

        raio_button_group = QButtonGroup(mainWindow)
        raio_button_group.addButton(ui.radioButton_continuous, 0)
        raio_button_group.addButton(ui.radioButton_trigger, 1)
        raio_button_group.buttonClicked.connect(radio_button_clicked)

        win_display_handles.append(ui.widget_display1.winId())
        win_display_handles.append(ui.widget_display2.winId())

        mainWindow.show()
        enum_devices()
        enable_ui_controls()

        sys.exit(app.exec_())
    finally:
        
        close_devices()
        MvCamera.MV_CC_Finalize()
