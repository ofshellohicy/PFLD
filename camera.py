# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print('pid: {}     GPU: {}'.format(os.getpid(),
                                   os.environ['CUDA_VISIBLE_DEVICES']))
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

#https://github.com/tensorflow/tensorflow/issues/24496
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

import numpy as np
import cv2
from mtcnn.detect_face import MTCNN


def main():
    meta_file = './models2/model4/model.meta'
    ckpt_file = './models2/model4/model.ckpt-61'
    image_size = 112
    with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:
            print('Loading feature extraction model.')
            saver = tf.compat.v1.train.import_meta_graph(meta_file)
            saver.restore(tf.compat.v1.get_default_session(), ckpt_file)

            graph = tf.compat.v1.get_default_graph()
            images_placeholder = graph.get_tensor_by_name('image_batch:0')
            phase_train_placeholder = graph.get_tensor_by_name('phase_train:0')

            landmarks = graph.get_tensor_by_name('pfld_inference/fc/BiasAdd:0')
            # landmark_L1 = graph.get_tensor_by_name('landmark_L1:0')
            # landmark_L2 = graph.get_tensor_by_name('landmark_L2:0')
            # landmark_L3 = graph.get_tensor_by_name('landmark_L3:0')
            # landmark_L4 = graph.get_tensor_by_name('landmark_L4:0')
            # landmark_L5 = graph.get_tensor_by_name('landmark_L5:0')
            # landmark_total = [landmark_L1, landmark_L2, landmark_L3, landmark_L4, landmark_L5]

            cap = cv2.VideoCapture(0)
            mtcnn = MTCNN()
            while True:
                ret, image = cap.read()
                height, width, _ = image.shape
                if not ret:
                    break
                boxes = mtcnn.predict(image)
                for box in boxes:
                    # score = box[4]
                    x1, y1, x2, y2 = (box[:4] + 0.5).astype(np.int32)
                    w = x2 - x1 + 1
                    h = y2 - y1 + 1

                    size = int(max([w, h]) * 1.1)
                    cx = x1 + w // 2
                    cy = y1 + h // 2
                    x1 = cx - size // 2
                    x2 = x1 + size
                    y1 = cy - size // 2
                    y2 = y1 + size

                    dx = max(0, -x1)
                    dy = max(0, -y1)
                    x1 = max(0, x1)
                    y1 = max(0, y1)

                    edx = max(0, x2 - width)
                    edy = max(0, y2 - height)
                    x2 = min(width, x2)
                    y2 = min(height, y2)

                    cropped = image[y1:y2, x1:x2]
                    if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
                        cropped = cv2.copyMakeBorder(cropped, dy, edy, dx, edx,
                                                     cv2.BORDER_CONSTANT, 0)
                    cropped = cv2.resize(cropped, (image_size, image_size))

                    input = cv2.resize(cropped, (image_size, image_size))
                    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
                    input = input.astype(np.float32) / 256.0
                    input = np.expand_dims(input, 0)

                    feed_dict = {
                        images_placeholder: input,
                        phase_train_placeholder: False
                    }
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0))
                    pre_landmarks = sess.run(landmarks, feed_dict=feed_dict)
                    pre_landmark = pre_landmarks[0]

                    pre_landmark = pre_landmark.reshape(-1, 2) * [size, size]
                    for (x, y) in pre_landmark.astype(np.int32):
                        cv2.circle(image, (x1 + x, y1 + y), 1, (0, 0, 255))
                cv2.imshow('0', image)
                if cv2.waitKey(10) == 27:
                    break


if __name__ == '__main__':
    main()
