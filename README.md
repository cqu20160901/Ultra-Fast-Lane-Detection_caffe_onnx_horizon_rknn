# UltraFastLaneDetection_caffe_onnx_horizon_rknn 

UltraFastLaneDetection 部署版本，后处理用python语言和C++语言形式进行改写，便于移植不同平台（caffe、onnx、rknn、Horizon）。

由于模型较大，无法放在仓库中，完整代码放在 UltraFastLaneDetection.zip。[完整代码] (https://github.com/cqu20160901/Ultra-Fast-Lane-Detection_caffe_onnx_horizon_rknn/releases/download/v1.0.0/UltraFastLaneDetection.zip)

# 文件夹结构说明
LaneDet_caffe：去除维度变换层的prototxt、caffeModel、测试图像、测试结果、测试demo脚本

LaneDet_onnx：onnx模型、测试图像、测试结果、测试demo脚本

LaneDet_rknn：rknn模型、测试（量化）图像、测试结果、onnx2rknn转换测试脚本

LaneDet_horizon：地平线模型、测试（量化）图像、测试结果、转换测试脚本、测试量化后onnx模型脚本

# 测试结果

![image](https://github.com/cqu20160901/Ultra-Fast-Lane-Detection_caffe_onnx_horizon_rknn/blob/main/LaneDet_caffe/test_result_caffe_zq.jpg)

# 参考链接
https://github.com/cfzd/Ultra-Fast-Lane-Detection

https://github.com/Jade999/caffe_lane_detection
