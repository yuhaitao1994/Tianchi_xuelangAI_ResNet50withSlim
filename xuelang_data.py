"""
纺织良品检测初赛第一阶段
使用TF-slim实现一个预训练的ResNet-50模型进行二分类
二分类的结果是有瑕疵/正常

此文件是数据处理文件

@author:Haitao Yu
"""
# -*- coding:utf-8


def get_data_and_label():
    """
    获取数据集和标签
    """
    return images, targets


def batch_generator(images, targets):
    """
    生成一个mini-batch
    Args:
    Returns:
        batch_images:
        batch_targets:
    """
    return batch_images, batch_labels


def out_excel(tmplist, outpath):
    print("----开始写入数据----")
    pos_data = xlsxwriter.Workbook(outpath)
    worksheet = pos_data.add_worksheet()
    # style = xlwt.XFStyle()
    # style.num_format_str = "[$-10804]0.00"
    rows_pos = len(tmplist)

    #print(rows_pos, col_pos)
    for i in range(rows_pos):
        worksheet.write(i, 0, tmplist[i])
    pos_data.close()
    print("----写入完成----")
