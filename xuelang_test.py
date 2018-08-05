"""
纺织良品检测初赛第一阶段
使用TF-slim实现一个预训练的ResNet-50模型进行二分类
二分类的结果是有瑕疵/正常

此文件是测试文件

@author:Haitao Yu
"""
# -*- coding:utf-8

import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import xuelang_classifier


flags = tf.flags.FLAGS
tf.flags.DEFINE_string(
    'model_ckpt_path', '/home/yuhaitao/code/xuelang_1/model/model.ckpt-480', 'path to checkpoint')
tf.flags.DEFINE_string(
    "test_image_path", "/home/yuhaitao/code/xuelang_1_data/xuelang_round1_test_b", "path to test image")


def main(_):
    """
    测试主函数
    """

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    with tf.Session(config=config) as sess:
        # 这个import_meta_graph函数是不只加载参数，也加载整个计算图，所以不用重新定义结构了
        saver = tf.train.import_meta_graph(flags.model_ckpt_path + ".meta")
        saver.restore(sess, flags.model_ckpt_path)
        # 通过张量名获取模型的数据入口和数据出口
        inputs = tf.get_default_graph().get_tensor_by_name("inputs:0")
        classes = tf.get_default_graph().get_tensor_by_name("classes:0")

        result = []

        for image_file in glob.glob(os.path.join(flags.test_image_path, "*.jpg")):
            # tf.image不能直接读取文件，先要对图片编码读取，然后解码成tensor，.eval()之后变成array形式
            image_read = tf.gfile.GFile(image_file, "rb").read()
            image = tf.image.decode_jpeg(image_read)
            image = tf.image.resize_image_with_crop_or_pad(image, 1000, 1000)
            image = tf.reshape(image, shape=[1, 1000, 1000, 3])

            probability = sess.run(classes, feed_dict={inputs: image.eval()})

            # 显示
            image_file = image_file[len(flags.test_image_path) + 1:]
            #probability = round(probability[0][1], 6)
            print(image_file, probability)
            result.append([image_file, probability])

        # 存储dataFrame
        df = pd.DataFrame(result, columns=["filename", "probability"])
        df.to_csv("xuelang_round1_submit_sample_20180709.csv", index=False)


if __name__ == '__main__':
    tf.app.run()
