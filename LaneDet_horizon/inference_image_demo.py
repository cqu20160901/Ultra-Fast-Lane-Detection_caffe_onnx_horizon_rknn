import logging
from horizon_tc_ui import HB_ONNXRuntime
from horizon_tc_ui.utils.tool_utils import init_root_logger
from math import exp
import cv2
import numpy as np



input_h = 288
input_w = 800

output_w = 200
sample_w = (input_w - 1) / (output_w - 1)


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



def preprocess(src_image):
    src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(src_image, (input_w, input_h))
    return img


def inference(model_path, image_path, input_layout, input_offset):
    # init_root_logger("inference.log", console_level=logging.INFO, file_level=logging.DEBUG)

    sess = HB_ONNXRuntime(model_file=model_path)
    sess.set_dim_param(0, 0, '?')

    if input_layout is None:
        logging.warning(f"input_layout not provided. Using {sess.layout[0]}")
        input_layout = sess.layout[0]

    src_img = cv2.imread(image_path)
    img_h, img_w = src_img.shape[:2]
    image_data = preprocess(src_img)

    # image_data = image_data.transpose((2, 0, 1))
    image_data = np.expand_dims(image_data, axis=0)

    input_name = sess.input_names[0]
    output_name = sess.output_names
    output = sess.run(output_name, {input_name: image_data}, input_offset=input_offset)

    print('inference finished, output len is:', len(output))

    result = postprocess(output[0][0])

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

    cv2.imwrite('./test_result_horizon_zq.jpg', src_img)


if __name__ == '__main__':
    print('This main ... ')

    model_path = './model_output/lane_quantized_model.onnx'
    image_path = './test.jpg'
    input_layout = 'NHWC'
    input_offset = 128

    inference(model_path, image_path, input_layout, input_offset)

