"""
纺织良品检测初赛第一阶段
使用TF-slim实现一个预训练的ResNet-50模型进行二分类
二分类的结果是有瑕疵/正常

此文件是对xml文件提取信息

@author:Haitao Yu
"""
# -*- coding:utf-8

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_read(path):
    """
    将xml文件中的瑕疵信息和位置读出
    Args:
        path:
    Return:
        xml_dataframe:
    """
    xml_list = []
    # xml文档由类似树的方式读取
    for xml_file in glob.glob(path + "/*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    image_path = os.path.join(
        "C:/Users/yuhai/Desktop/python_practice/xuelangAI_data/Train/瑕疵/边白印")
    xml_df = xml_read(image_path)
    print(xml_df)
    #xml_df.to_csv('road_signs_labels.csv', index=None)
    #print('Successfully converted xml to csv.')


if __name__ == "__main__":
    main()
