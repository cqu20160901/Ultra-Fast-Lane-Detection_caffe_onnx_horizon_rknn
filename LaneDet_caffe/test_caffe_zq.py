import numpy as np
import cv2
from math import exp

import sys

caffe_root = '/zhangqian/caffe-master/'
sys.path.insert(0, caffe_root + 'python')
import caffe

net = caffe.Net('./lane.prototxt', './lane.caffemodel', caffe.TEST)


image_mean = np.array([0.485, 0.456, 0.406])
image_std = np.array([0.229, 0.224, 0.225])

input_h = 288
input_w = 800

output_w = 200
sample_w = (input_w - 1) / (output_w - 1)


def preprocess(src_img):
    img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_w, input_h))
    img = img / 255.
    img = (img - image_mean) / image_std
    return img


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



def inference(img_path):
    src_img = cv2.imread(img_path)
    img_h, img_w = src_img.shape[0:2]
    img = preprocess(src_img)

    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['input'].data[...] = img
    output = net.forward()
    res = output['output']

    result = postprocess(res[0][0])

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

    cv2.imwrite('./test_result_caffe_zq.jpg', src_img)


if __name__ == '__main__':
    print('This is main ....')
    img_path = 'test.jpg'
    inference(img_path)
