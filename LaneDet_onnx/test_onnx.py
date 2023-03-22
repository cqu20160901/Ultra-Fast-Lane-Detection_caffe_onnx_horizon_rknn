import numpy as np
import onnxruntime as ort
import cv2
import scipy.special

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


def inference(img_path):
    src_img = cv2.imread(img_path)

    img_h, img_w = src_img.shape[:2]
    img = preprocess(src_img)

    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    img = np.expand_dims(img, axis=0)

    ort_session = ort.InferenceSession('./lane.onnx')
    res = (ort_session.run(None, {'input_input': img}))

    out_j = res[0][0]
    out_j = out_j[:, ::-1, :]

    prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)

    idx = np.arange(output_w) + 1
    idx = idx.reshape(-1, 1, 1)

    loc = np.sum(prob * idx, axis=0)
    out_j = np.argmax(out_j, axis=0)
    loc[out_j == output_w] = 0

    out_j = loc

    print(out_j.shape)

    for i in range(out_j.shape[1]):
        if np.sum(out_j[:, i] != 0) > 2:

            for k in range(out_j.shape[0]):
                if out_j[k, i] > 0:
                    point = (int(out_j[k, i] * sample_w * img_w / input_w) - 1, int(img_h - k * 20) - 1)
                    if i == 0:
                        cv2.circle(src_img, point, 5, (255, 0, 0), -1)
                    if i == 1:
                        cv2.circle(src_img, point, 5, (0, 255, 0), -1)
                    if i == 2:
                        cv2.circle(src_img, point, 5, (0, 0, 255), -1)
                    if i == 3:
                        cv2.circle(src_img, point, 5, (0, 255, 255), -1)
    cv2.imwrite('./test_result_onnx.jpg', src_img)


if __name__ == '__main__':
    print('This is main .... ')
    img_path = './test.jpg'
    inference(img_path)
