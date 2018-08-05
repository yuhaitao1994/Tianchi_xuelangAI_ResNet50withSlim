"""
纺织良品检测初赛第一阶段
使用TF-slim实现一个预训练的ResNet-50模型进行二分类
二分类的结果是有瑕疵/正常

此文件是模型文件

@author:Haitao Yu
"""
# -*- coding:utf-8

import tensorflow as tf
from tensorflow.contrib.slim import nets

slim = tf.contrib.slim


class xuelang_classifier(object):
    """
    分类器模型
    """

    def __init__(self, is_training, num_classes):
        """
        初始化
        Args:
            is_training:bool型，区分模型是训练还是测试
            num_classes:整形，类别数
        """
        self._is_training = is_training
        self._num_classes = num_classes

    def num_classes(self):
        return self._num_classes

    def inputs_process(self, inputs):
        """
        对输入数据进行处理
        Args:
            inputs:一个(batch_size*长*宽*图像通道数)的四维tensor
        Returns:
            preprocessed_inputs:一个同样大小的经过预处理的tenssor
        """
        preprocessed_inputs = tf.to_float(inputs)
        preprocessed_inputs = tf.subtract(preprocessed_inputs, 128.0)
        preprocessed_inputs = tf.div(preprocessed_inputs, 128.0)
        return preprocessed_inputs

    def networks(self, preprocessed_inputs):
        """
        构造卷积层和全连接层，输出的结果提供给softmax和loss层分别计算
        在初赛，使用预训练的ResNet-50
        Args:
            preprocessed_inputs:
        Returns:
            prediction_dict
        """
        net, endpoints = nets.resnet_v2.resnet_v2_50(
            preprocessed_inputs, num_classes=None, is_training=self._is_training)
        net = tf.squeeze(net, axis=[1, 2])
        net = slim.dropout(net, keep_prob=0.5, scope='scope')
        net = slim.fully_connected(
            net,
            num_outputs=self._num_classes,
            activation_fn=None,
            scope="resnet_fully_connected")
        prediction_dict = {'logits': net}

        return prediction_dict

    def softMax(self, prediction_dict):
        """
        后期处理层，用softmax分类器坐作最后的分类,这步的用处主要是得出分类值
        Args:
            prediction_dict:
        Returns:
            final_dict
        """
        logits = prediction_dict['logits']
        logits = tf.nn.softmax(logits, name="softmax")
        classes = tf.cast(tf.argmax(logits, axis=1),
                          dtype=tf.int32, name="classes")
        final_dict = {'classes': classes}
        # final_dict_ = tf.identity(final_dict, name="classes")

        return final_dict

    def loss(self, prediction_dict, truth_list):
        """
        计算损失的层
        Args:
            prediction_dict:
            truth_list:
        Returns:
            loss_dict:
        """
        logits = prediction_dict['logits']
        # “硬”softmax代价函数
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=truth_list))
        loss_dict = {'loss': loss}
        return loss_dict

    def accuracy(self, final_dict, truth_list):
        """
        计算准确率
        Args:
            final_dict:
            truth_list:
        Returns:
            accuracy:
        """
        classes = final_dict['classes']
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(classes, truth_list), dtype=tf.float32))
        return accuracy
