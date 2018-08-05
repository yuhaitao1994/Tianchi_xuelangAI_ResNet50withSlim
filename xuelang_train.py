"""
纺织良品检测初赛第一阶段
使用TF-slim实现一个预训练的ResNet-50模型进行二分类
二分类的结果是有瑕疵/正常

此文件是训练文件

@author:Haitao Yu
"""
# -*- coding:utf-8

import numpy as np
import tensorflow as tf
import os
import xuelang_classifier
slim = tf.contrib.slim

from xuelang_generateRecord import get_record_dataset

flags = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', 8, 'number of images in on batch')
tf.flags.DEFINE_float('start_learnig_rate', 0.001,
                      'learning rate at the beginning')
tf.flags.DEFINE_integer('epochs', 100, 'number of training epoches')
tf.flags.DEFINE_string(
    'resnet_path', '/home/yuhaitao/code/xuelang_1/model/resnet_v2_50.ckpt', 'the path of resnet')
tf.flags.DEFINE_string('model_output_path', '/home/yuhaitao/code/xuelang_1/model/model.ckpt',
                       'the path to save model')
tf.flags.DEFINE_string(
    "record_path", "/home/yuhaitao/code/xuelang_1_data/record/image.record", "Path to output")


def main(_):
    """
    训练主函数
    """
    """
    从tfrecord恢复数据
    """
    dataset = get_record_dataset(
        flags.record_path, num_samples=2022, num_classes=2)
    data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
    image, label = data_provider.get(['image', 'label'])
    # 数据增强
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize_image_with_crop_or_pad(image, 1000, 1000)

    batch_images, batch_labels = tf.train.shuffle_batch([image, label], batch_size=flags.batch_size,
                                                        capacity=1500, min_after_dequeue=1000,
                                                        allow_smaller_final_batch=True)

    """
    定义占位符等模型必要结构
    """
    inputs = tf.placeholder(
        tf.float32, shape=[None, 1000, 1000, 3], name="inputs")
    labels = tf.placeholder(tf.int32, shape=[None], name="labels")

    model = xuelang_classifier.xuelang_classifier(
        is_training=True, num_classes=2)
    preprocessed_inputs = model.inputs_process(inputs)
    prediction_dict = model.networks(preprocessed_inputs)
    loss_dict = model.loss(prediction_dict, labels)
    loss = loss_dict['loss']
    # softmax输出类别
    final_dict = model.softMax(prediction_dict)

    # 训练集的准确率
    train_acc = model.accuracy(final_dict, labels)

    # 需要理解一下global_step有什么用
    # gloabal_step代表全局步数，随着optimizer自加
    global_step = tf.Variable(0, trainable=False)
    # 学习率衰减
    learning_rate = tf.train.exponential_decay(
        learning_rate=0.001, global_step=global_step, decay_steps=150, decay_rate=0.95)

    # 优化器
    optimizer = tf.train.AdamOptimizer(learning_rate, beta2=0.99)
    train_step = optimizer.minimize(loss, global_step)

    # 参数初始化
    init = tf.global_variables_initializer()

    """
    导入预训练的resnet,首先要去掉预训练里面的一些不导入的变量
    """
    """
    exclusions = ["Logits", "biases", "resnet_fully_connected/weights"]
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.endswith(exclusion):
                excluded = True
        if not excluded:
            variables_to_restore.append(var)

    saver_restore = tf.train.Saver(var_list=variables_to_restore)
    """

    # 定义存储模型
    saver = tf.train.Saver(tf.global_variables())

    """
    开始计算图
    """
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    with sess:
        sess.run(init)
        # 读取预训练模型参数
        #saver_restore.restore(sess, flags.resnet_path)

        # 定义读取数据的线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # 定义两个loss列表
        total_loss = []
        epoch_loss = []
        # 定义训练集的准确率
        training_accuracy = []
        max_train_acc = 0
        iterations = 240
        # 开始迭代
        for i in range(flags.epochs):
            # 在循环前先定义一些flag
            one_epoch_loss = 0
            one_epoch_acc = 0
            # 开始进行每次iteration的循环
            for j in range(iterations):

                feed = {inputs: batch_images.eval(), labels: batch_labels.eval()}

                batch_loss, batch_acc, _, Global_step = sess.run(
                    [loss, train_acc, train_step, global_step], feed_dict=feed)

                one_epoch_loss += batch_loss
                one_epoch_acc += batch_acc

                total_loss.append(batch_loss)

                print('global_step:{},batch_loss:{:.6f},:batch_acc:{:.6f}'.format(
                    Global_step, batch_loss, batch_acc))

            print('epoch: {}/{}... '.format(i + 1, flags.epochs),
                  'loss: {:.6f}... '.format(one_epoch_loss / iterations),
                  'accuracy:{:.4f}...'.format(one_epoch_acc / iterations))
            epoch_loss.append(one_epoch_loss / iterations)
            training_accuracy.append(one_epoch_acc / iterations)

            # 保存模型
            if training_accuracy[i] > max_train_acc:
                max_train_acc = training_accuracy[i]
                saver.save(sess, flags.model_output_path,
                           global_step=global_step)

        # 记录loss数据
        """
        xuelang_data.out_excel(total_loss, "total_loss.xlsx")
        xuelang_data.out_excel(epoch_loss, "epoch_loss.xlsx")
        xuelang_data.out_excel(train_acc, "training_acc.xlsx")"""

        coord.request_stop()
        coord.join(threads)

    sess.close()


if __name__ == "__main__":
    tf.app.run()
