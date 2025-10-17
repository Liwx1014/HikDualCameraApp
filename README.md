## 项目说明
本项目基于海康工业相机SDK,基于Python接口开发了双目相机采集客户端
![alt text](image-6.png)

## 项目功能
- [x] 支持任意数量相机的连接
- [x] 触发模式支持软同步、硬同步
- [x] 软同步触发保存图像到本地，采集误差1ms
- [x] 帧率、增益、曝光时间设置
- [x] Pyinstaller打包发布
- [x] 在线、离线标定，支持单目和双目内外参计算，标定过程可视化
- [x] 双目矫正

## 代码说明
- [双目在线标定代码](calibrate_dua.py)
- [双目USB模组在线标定代码](calibrate_sig.py)  
- [离线标定](calibrate_local.py)
- [矫正](stereo_rectification.py)
- [双目测距]()

## 参考
- [Matlab标定](https://blog.csdn.net/weixin_43956351/article/details/94394892)
- [参考项目](https://github.com/TemugeB/python_stereo_camera_calibrate#)
- [其他标定工具](https://github.com/ethz-asl/kalibr)
  
## 注意事项
标定是整个测距的基石，在采集图像时一定要保证图像清晰选取大的、坚硬棋盘格、多种角度（倾斜、反转、旋转，不同深度）下采集，这样计算出来的内参足够鲁棒性。
详细介绍移步我的[博客](https://blog.csdn.net/Colin_xuan?type=blog)

## 海康相机技术文档
- [官方技术说明](https://www.hikrobotics.com/cn/machinevision/visionproduct?typeId=27&id=249&pageNumber=1&pageSize=20&showEol=false)
- [硬触发技术文档](开发说明.md)





