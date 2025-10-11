# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'PyUIMultipleCameras.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: This file has been manually modified to improve the UI layout and apply a custom theme.
# Re-generating it from a .ui file will overwrite these changes.

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QStyle, QGroupBox, QFormLayout, QVBoxLayout
import os

class Ui_MainWindow(object):
    
    # --- UI THEME STYLESHEET (OPTIMIZED) ---
    dark_green_stylesheet = """
        QMainWindow, QWidget {
            background-color: #1E3A3A; /* Very dark desaturated green */
            color: #F0F0F0;
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 10pt;
        }
        QGroupBox {
            background-color: #27443E; /* Dark green container */
            border: 1px solid #3A6055;
            border-radius: 8px;
            font-weight: bold;
            /* --- MODIFICATION START --- */
            margin-top: 15px; /* Increased from 10px to give the title more vertical space */
            padding: 30px 12px 12px 12px; /* Top padding increased from 20px to 30px to push content down */
            /* --- MODIFICATION END --- */
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 3px 12px; /* Slightly increased padding for a better look */
            background-color: #3A6055;
            border-radius: 4px;
            color: #FFFFFF;
        }
        QPushButton, QComboBox {
            background-color: #008060; /* Bright, saturated green */
            color: white;
            border: none;
            border-radius: 5px;
            padding: 8px 12px;
        }
        QPushButton:hover, QComboBox:hover {
            background-color: #009A72;
        }
        QPushButton:pressed {
            background-color: #00664D;
        }
        QPushButton:disabled, QComboBox:disabled {
            background-color: #556B66;
            color: #AAAAAA;
        }
        QComboBox::drop-down {
            border: none;
        }
        QComboBox QAbstractItemView {
            background-color: #27443E;
            border: 1px solid #3A6055;
            color: #F0F0F0;
            selection-background-color: #008060;
        }
        QLineEdit, QTextEdit {
            background-color: #1A2F2F;
            border: 1px solid #3A6055;
            border-radius: 4px;
            padding: 5px;
            color: #F0F0F0;
        }
        QTextEdit {
            font-family: 'Consolas', 'Courier New', monospace;
        }
        QLabel {
            color: #F0F0F0;
        }
        QRadioButton {
            spacing: 8px;
            color: #F0F0F0;
        }
        QRadioButton::indicator {
            border: 1px solid #3A6055;
            width: 16px;
            height: 16px;
            border-radius: 9px;
            background-color: #1A2F2F;
        }
        QRadioButton::indicator:checked {
            background-color: #008060;
            border: 1px solid #009A72;
        }
        QStatusBar {
            background-color: #27443E;
            color: #F0F0F0;
        }
        QMenuBar {
            background-color: #27443E;
            color: #F0F0F0;
        }
    """

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 850)
        MainWindow.setMinimumSize(QtCore.QSize(1100, 550))
        
        MainWindow.setStyleSheet(self.dark_green_stylesheet)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(current_dir, 'logo.ico')
        app_icon = QtGui.QIcon()
        app_icon.addFile(icon_path, QtCore.QSize(32,32))
        MainWindow.setWindowIcon(app_icon)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        self.main_h_layout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.main_h_layout.setContentsMargins(15, 15, 15, 15)
        self.main_h_layout.setSpacing(15)
        self.main_h_layout.setObjectName("main_h_layout")

        self.left_v_layout = QtWidgets.QVBoxLayout()
        self.left_v_layout.setSpacing(15)
        self.left_v_layout.setObjectName("left_v_layout")
        
        self.display_grid_layout = QtWidgets.QGridLayout()
        self.display_grid_layout.setSpacing(15)
        self.display_grid_layout.setObjectName("display_grid_layout")
        
        self.checkBox_1 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_1.setObjectName("checkBox_1")
        self.display_grid_layout.addWidget(self.checkBox_1, 0, 0, 1, 1)
        self.widget_display1 = QtWidgets.QWidget(self.centralwidget)
        self.widget_display1.setMinimumSize(QtCore.QSize(320, 240))
        self.widget_display1.setStyleSheet("background-color: black; border-radius: 8px;")
        self.widget_display1.setObjectName("widget_display1")
        self.display_grid_layout.addWidget(self.widget_display1, 1, 0, 1, 1)

        self.checkBox_2 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_2.setObjectName("checkBox_2")
        self.display_grid_layout.addWidget(self.checkBox_2, 0, 1, 1, 1)
        self.widget_display2 = QtWidgets.QWidget(self.centralwidget)
        self.widget_display2.setMinimumSize(QtCore.QSize(320, 240))
        self.widget_display2.setStyleSheet("background-color: black; border-radius: 8px;")
        self.widget_display2.setObjectName("widget_display2")
        self.display_grid_layout.addWidget(self.widget_display2, 1, 1, 1, 1)
        
        self.left_v_layout.addLayout(self.display_grid_layout)
        
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setMaximumSize(QtCore.QSize(16777215, 120))
        self.textEdit.setReadOnly(True)
        self.textEdit.setObjectName("textEdit")
        self.left_v_layout.addWidget(self.textEdit)
        
        self.main_h_layout.addLayout(self.left_v_layout)
        
        self.right_v_layout = QVBoxLayout()
        self.right_v_layout.setContentsMargins(0, 0, 0, 0)
        self.right_v_layout.setSpacing(25)
        self.right_v_layout.setObjectName("right_v_layout")

        self.group_device_connection = QGroupBox("1. 设备连接")
        self.group_device_connection.setObjectName("group_device_connection")
        self.layout_device_connection = QtWidgets.QGridLayout(self.group_device_connection)
        self.layout_device_connection.setVerticalSpacing(10)
        self.pushButton_enum = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_open = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_close = QtWidgets.QPushButton(self.centralwidget)
        self.layout_device_connection.addWidget(self.pushButton_open, 0, 0, 1, 1)
        self.layout_device_connection.addWidget(self.pushButton_close, 0, 1, 1, 1)
        self.layout_device_connection.addWidget(self.pushButton_enum, 1, 0, 1, 2)
        self.right_v_layout.addWidget(self.group_device_connection)

        self.group_trigger = QGroupBox("2. 触发设置")
        self.group_trigger.setObjectName("group_trigger")
        self.layout_trigger = QVBoxLayout(self.group_trigger)
        self.layout_trigger.setSpacing(15)
        self.layout_trigger_radio = QtWidgets.QHBoxLayout()
        self.radioButton_continuous = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_trigger = QtWidgets.QRadioButton(self.centralwidget)
        self.layout_trigger_radio.addWidget(self.radioButton_continuous)
        self.layout_trigger_radio.addWidget(self.radioButton_trigger)
        self.layout_trigger.addLayout(self.layout_trigger_radio)
        
        self.layout_trigger_source = QtWidgets.QHBoxLayout()
        self.label_triggerSource = QtWidgets.QLabel("触发源:", self.centralwidget)
        self.comboBox_triggerSource = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_triggerSource.addItems(["Software", "Line0"])
        self.layout_trigger_source.addWidget(self.label_triggerSource)
        self.layout_trigger_source.addWidget(self.comboBox_triggerSource)
        self.layout_trigger_source.setStretch(1, 1)
        self.layout_trigger.addLayout(self.layout_trigger_source)
        self.right_v_layout.addWidget(self.group_trigger)
        
        self.group_acquisition = QGroupBox("3. 采集控制")
        self.group_acquisition.setObjectName("group_acquisition")
        self.layout_acquisition = QtWidgets.QGridLayout(self.group_acquisition)
        self.layout_acquisition.setVerticalSpacing(10)
        self.pushButton_startGrab = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_stopGrab = QtWidgets.QPushButton(self.centralwidget)
        self.layout_acquisition.addWidget(self.pushButton_startGrab, 0, 0, 1, 1)
        self.layout_acquisition.addWidget(self.pushButton_stopGrab, 0, 1, 1, 1)
        self.right_v_layout.addWidget(self.group_acquisition)

        self.group_params = QGroupBox("4. 参数设置")
        self.group_params.setObjectName("group_params")
        self.layout_params = QFormLayout(self.group_params)
        self.layout_params.setLabelAlignment(QtCore.Qt.AlignLeft)
        self.layout_params.setVerticalSpacing(10)
        self.label_exposure = QtWidgets.QLabel("曝光时间(us):", self.centralwidget)
        self.lineEdit_exposureTime = QtWidgets.QLineEdit(self.centralwidget)
        self.label_gain = QtWidgets.QLabel("增益:", self.centralwidget)
        self.lineEdit_gain = QtWidgets.QLineEdit(self.centralwidget)
        self.label_frameRate = QtWidgets.QLabel("帧率(fps):", self.centralwidget)
        self.lineEdit_frameRate = QtWidgets.QLineEdit(self.centralwidget)
        self.layout_params.addRow(self.label_exposure, self.lineEdit_exposureTime)
        self.layout_params.addRow(self.label_gain, self.lineEdit_gain)
        self.layout_params.addRow(self.label_frameRate, self.lineEdit_frameRate)
        self.pushButton_setParams = QtWidgets.QPushButton("应用参数", self.centralwidget)
        self.layout_params.addRow(self.pushButton_setParams)
        self.right_v_layout.addWidget(self.group_params)
        
        self.right_v_layout.addStretch(1) 
        
        self.pushButton_triggerOnce = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_saveImg = QtWidgets.QPushButton(self.centralwidget)
        self.right_v_layout.addWidget(self.pushButton_triggerOnce)
        self.right_v_layout.addWidget(self.pushButton_saveImg)
        self.main_h_layout.addLayout(self.right_v_layout)
        self.main_h_layout.setStretch(0, 3) 
        self.main_h_layout.setStretch(1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1200, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "双目相机采集系统"))
        
        self.checkBox_1.setText(_translate("MainWindow", "相机 1"))
        self.checkBox_2.setText(_translate("MainWindow", "相机 2"))
        
        self.pushButton_enum.setText(_translate("MainWindow", "搜索设备"))
        self.pushButton_open.setText(_translate("MainWindow", "打开设备"))
        self.pushButton_close.setText(_translate("MainWindow", "关闭设备"))
        self.pushButton_startGrab.setText(_translate("MainWindow", "开始采集"))
        self.pushButton_stopGrab.setText(_translate("MainWindow", "停止采集"))
        self.pushButton_saveImg.setText(_translate("MainWindow", "保存图片"))
        self.pushButton_triggerOnce.setText(_translate("MainWindow", "软触发一次"))
        
        self.radioButton_continuous.setText(_translate("MainWindow", "连续模式"))
        self.radioButton_trigger.setText(_translate("MainWindow", "触发模式"))
        self.pushButton_setParams.setText(_translate("MainWindow", "应用参数"))