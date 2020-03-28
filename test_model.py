from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print('pid: {}     GPU: {}'.format(os.getpid(),
                                   os.environ['CUDA_VISIBLE_DEVICES']))
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

from generate_data import gen_data


def main():
    # meta_file = './models2/model0/model.meta'
    # ckpt_file = './models2/model0/model.ckpt-0'
    meta_file = './models1/model_test/model.meta'
    ckpt_file = './models1/model_test/model.ckpt-13'
    # test_list = './data/300w_image_list.txt'

    image_size = 112

    image_files = 'data/test_data/list.txt'
    out_dir = 'result'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:
            print('Loading feature extraction model.')
            saver = tf.compat.v1.train.import_meta_graph(meta_file)
            saver.restore(tf.compat.v1.get_default_session(), ckpt_file)

            graph = tf.compat.v1.get_default_graph()
            images_placeholder = graph.get_tensor_by_name('image_batch:0')
            phase_train_placeholder = graph.get_tensor_by_name('phase_train:0')

            # landmark_L1 = graph.get_tensor_by_name('landmark_L1:0')
            # landmark_L2 = graph.get_tensor_by_name('landmark_L2:0')
            # landmark_L3 = graph.get_tensor_by_name('landmark_L3:0')
            # landmark_L4 = graph.get_tensor_by_name('landmark_L4:0')
            # landmark_L5 = graph.get_tensor_by_name('landmark_L5:0')
            # landmark_total = [
            #     landmark_L1, landmark_L2, landmark_L3, landmark_L4, landmark_L5
            # ]
            landmark_total = graph.get_tensor_by_name(
                'pfld_inference/fc/BiasAdd:0')

            file_list, train_landmarks, train_attributes, train_euler_angles = gen_data(
                image_files)
            print(file_list)
            for file in file_list:
                filename = os.path.split(file)[-1]
                image = cv2.imread(file)
                # image = cv2.resize(image, (image_size, image_size))
                input = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
                input = cv2.resize(input, (image_size, image_size))
                input = input.astype(np.float32) / 256.0
                input = np.expand_dims(input, 0)
                # (1x112x112x3)
                # print(input.shape)

                feed_dict = {
                    images_placeholder: input,
                    phase_train_placeholder: False
                }

                pre_landmarks = sess.run(landmark_total, feed_dict=feed_dict)
                # 98x2
                # print(pre_landmarks)
                pre_landmark = pre_landmarks[0]

                h, w, _ = image.shape
                pre_landmark = pre_landmark.reshape(-1, 2) * [h, w]
                for (x, y) in pre_landmark.astype(np.int32):
                    cv2.circle(image, (x, y), 1, (0, 0, 255))
                cv2.imshow('0', image)
                cv2.waitKey(0)
                cv2.imwrite(os.path.join(out_dir, filename), image)


if __name__ == '__main__':
    main()
