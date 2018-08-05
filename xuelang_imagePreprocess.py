"""
纺织良品检测初赛第一阶段
使用TF-slim实现一个预训练的ResNet-50模型进行二分类
二分类的结果是有瑕疵/正常

此文件是生成对图片进行预处理

@author:Haitao Yu
"""
# -*- coding:utf-8

from random import normalvariate, randint
import numpy as np
import os
from PIL import Image

from xuelang_xmlProcess import xml_read


def rescale(image):
    """
    等比例缩放2560*1920的图片,短边的缩放范围是(240,480)
    Args:
        image:
    Returns:
        image.resize:
    """
    w = image.size[0]
    h = image.size[1]
    sizeMax = 1500
    sizeMin = 1200
    random_size = randint(sizeMin, sizeMax)
    if w < h:
        return image.resize((sizeMin, round(h / w * sizeMin)))
    else:
        return image.resize((round(w / h * sizeMin), sizeMin))


def random_crop(image):
    """
    随机裁剪图片到224*224大小，该function只针对正常图片
    Args:
        image:
    Returns:
        image.crop:
    """
    w = image.size[0]
    h = image.size[1]
    size = 1000
    new_left = randint(0, w - size)
    new_upper = randint(0, h - size)
    return image.crop((new_left, new_upper, size + new_left, size + new_upper))


def detect_crop(image, xml_dict):
    """
    对有瑕疵的图片，按照xml的瑕疵位置裁剪,并且不对图片缩放
    Args:
        image:
        xml_dict:字典类型，携带一个瑕疵的信息
    Returns:
        image.crop:
    """
    w = image.size[0]
    h = image.size[1]
    size = 224
    # 瑕疵区域的中心点
    centerW = (xml_dict["xmax"] + xml_dict["xmin"]) / 2
    centerH = h - (xml_dict["ymax"] + xml_dict["ymin"]) / 2
    if centerW < (w - centerW):
        new_left = randint(max(0, centerW - 168), centerW)
        if centerH < (h - centerH):
            new_upper = randint(max(0, centerH - 168), centerH)
        else:
            new_upper = randint(centerH, min(centerH + 168, h)) - size
    else:
        new_left = randint(centerW, min(centerW + 168, h)) - size
        if centerH < (h - centerH):
            new_upper = randint(max(0, centerH - 168), centerH)
        else:
            new_upper = randint(centerH, min(centerH + 168, h)) - size

    return image.crop(new_left, new_upper, new_left + size, new_upper + size)


def horizontal_flip(image):
    """
    水平翻转图片
    """
    return image.transpose(Image.FLIP_LEFT_RIGHT)


def main():
    image = Image.open(
        "C:/Users/yuhai/Desktop/python_practice/xuelangAI_data/train_data/瑕疵/扎洞/J01_2018.06.13 13_58_14.jpg")
    image = rescale(image)
    #image = random_crop(image)
    image.show()


if __name__ == "__main__":
    main()
