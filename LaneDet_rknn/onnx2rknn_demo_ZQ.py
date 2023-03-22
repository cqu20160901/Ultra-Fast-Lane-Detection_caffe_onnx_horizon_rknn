import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN
from math import exp

ONNX_MODEL = './lane.onnx'
RKNN_MODEL = './lane.rknn'
DATASET = './dataset.txt'

QUANTIZE_ON = True


input_h = 288
input_w = 800

output_w = 200
sample_w = (input_w - 1) / (output_w - 1)


def export_rknn_inference(img):
    # Create RKNN object
    rknn = RKNN(verbose=False)

    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[123, 114, 106]], std_values=[[58, 57, 57]], quantized_algorithm='normal', quantized_method='channel', target_platform='rk3588')
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL, outputs=['output'])
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET, rknn_batch_size=1)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    # ret = rknn.init_runtime(target='rk3566')
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    rknn.release()
    print('done')

    return outputs


def postprocess(output):
    result = np.zeros(shape=(18, 4))
    for i in range(4):
        for j in range(18):
            total = 0
            maxvalue = 0
            maxindex = 0
            for k in range(200):
                if maxvalue < output[k, j, i]:
                    maxvalue = output[k, j, i]
                    maxindex = k
                if k == 199:
                    if maxvalue < output[k + 1, j, i]:
                        maxvalue = output[k + 1, j, i]
                        maxindex = k

                tmp = exp(output[k, j, i])
                total += tmp

            for k in range(200):
                if maxindex < 199:
                    tmp = exp(output[k, j, i]) / total
                    output[k, j, i] = tmp
                    result[17 - j, i] += tmp * (k + 1)

    return result


if __name__ == '__main__':
    print('This is main ...')
    
    img_path = './test.jpg'
    src_img = cv2.imread(img_path)
    
    img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    img_h, img_w = img.shape[:2]
    
    img = cv2.resize(img, (800, 288))
    img = np.expand_dims(img, 0)

    outputs = export_rknn_inference(img)
    print(outputs[0].shape)
    
    result = postprocess(outputs[0][0])

    for i in range(result.shape[1]):
        for k in range(result.shape[0]):
            if result[k, i] > 0:
                point = (int(result[k, i] * sample_w * img_w / input_w) - 1, int(img_h - k * 20) - 1)
                if i == 0:
                    cv2.circle(src_img, point, 5, (255, 0, 0), -1)
                if i == 1:
                    cv2.circle(src_img, point, 5, (0, 255, 0), -1)
                if i == 2:
                    cv2.circle(src_img, point, 5, (0, 0, 255), -1)
                if i == 3:
                    cv2.circle(src_img, point, 5, (0, 255, 255), -1)

    cv2.imwrite('./test_result_rknn_zq.jpg', src_img)

