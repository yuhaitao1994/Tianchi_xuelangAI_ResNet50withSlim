"""
纺织良品检测初赛第一阶段
使用TF-slim实现一个预训练的ResNet-50模型进行二分类
二分类的结果是有瑕疵/正常

此文件是生成TFRecord

@author:Haitao Yu
"""
# -*- coding:utf-8

import glob
import io
import os
import tensorflow as tf
#import matplotlib.pyplot as plt
slim = tf.contrib.slim

from PIL import Image

flags = tf.flags.FLAGS
tf.flags.DEFINE_string(
    "image_path", "/home/yuhaitao/code/xuelang_1_data/train_data", 'Path to images')
tf.flags.DEFINE_string(
    "output_path", "/home/yuhaitao/code/xuelang_1_data/record/image.record", "Path to output")

"""
tf.train.Feature类的作用是将图像编码为字符或数字特征
"""


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_tf_example(image_path):
    """
    将特征写入协议缓冲区
    """
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)

    width, height = image.size
    # 这里要通过xml阅读的信息得到label,初赛简单起见只分0，1label
    if "瑕疵" in image_path:
        label = 1
    else:
        label = 0

    tf_example = tf.train.Example(
        features=tf.train.Features(feature={
            'image/encoded': bytes_feature(encoded_jpg),
            'image/format': bytes_feature('jpg'.encode()),
            'image/class/label': int64_feature(label),
            'image/height': int64_feature(height),
            'image/wigth': int64_feature(width)}))
    return tf_example


def generate_tfRecord(image_path, output_path):
    """
    将数据写入.record文件中
    """
    writer = tf.python_io.TFRecordWriter(output_path)
    # 遍历所有文件夹
    for lists in os.listdir(image_path):
        if lists == "瑕疵":
            xiaci = os.path.join(image_path, lists)
            for dirs in os.listdir(xiaci):
                path = os.path.join(xiaci, dirs)
                # 将目录下的图片都做成record
                for image_file in glob.glob(os.path.join(path, "*.jpg")):
                    tf_example = create_tf_example(image_file)
                    writer.write(tf_example.SerializeToString())
        else:
            zhengchang = os.path.join(flags.image_path, lists)
            # 将目录下的图片都做成record
            for image_file in glob.glob(os.path.join(zhengchang, "*.jpg")):
                tf_example = create_tf_example(image_file)
                writer.write(tf_example.SerializeToString())

    writer.close()


def get_record_dataset(record_path, reader=None,
                       image_shape=[2560, 1920, 3], num_samples=1,
                       num_classes=2):
    """
    读取tf.record中的文件
    Args:
        recode_path:
        reader:
        image_shape:
        num_samples:
        num_classes:
    Returns:
        slim.dataset:
    """
    if not reader:
        reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/class/label': tf.FixedLenFeature([1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64))
    }

    items_to_handlers = {
        # image_key='image/encoded',# format_key='image/format',
        'image': slim.tfexample_decoder.Image(shape=image_shape, channels=3),
        'label': slim.tfexample_decoder.Tensor('image/class/label', shape=[])
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None
    items_to_descriptions = {
        'image': 'An image with shape image_shape.',
        'label': 'A single integer between 0 and 1.'
    }

    # 返回一个slim的标准dataset
    return slim.dataset.Dataset(
        data_sources=record_path,
        reader=reader,
        decoder=decoder,
        num_samples=num_samples,
        num_classes=num_classes,
        items_to_descriptions=items_to_descriptions,
        labels_to_names=labels_to_names
    )


def main():
    # 写入
    #generate_tfRecord(flags.image_path, flags.output_path)

    num_samples = 0
    for record in tf.python_io.tf_record_iterator(flags.output_path):
        num_samples += 1
    print(num_samples)

    # 读取
    dataset = get_record_dataset(flags.output_path, num_samples=1)
    # 这个函数默认每次读取一个数据
    data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
    image, label = data_provider.get(['image', 'label'])
    print(label)
    print(image)

    # 尝试将tensor形式的image改回图像形式
    image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
    print(image)
    """with tf.Session() as sess:
        plt.figure()
        plt.imshow(image.eval())
        plt.savefig("aaa.jpg")"""


if __name__ == '__main__':
    main()
