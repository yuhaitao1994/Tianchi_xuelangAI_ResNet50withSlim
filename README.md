# Tianchi_xuelangAI_ResNet50withSlim

___本项目是天池大数据竞赛-雪浪AI布匹瑕疵检测比赛初赛的代码。___  
___因为参加时间已经临近初赛结尾，所以只是实现了代码功能，没有仔细调参。可以作为经验在以后的比赛中深入学习。___  

一开始在天池网站上看到这个比赛的时候已经接近尾声了，所以只是调通了整体代码，并没有深入研究模型调参与具体方法。这个
比赛是布匹瑕疵检测，基本上就是图像的目标检测任务。但是初赛只要求做到分类就可以，所以只是简单地将模型写成了二分类。整个项目采用
了tensorflow的slim模块加载预训练的ResNetV2-50模型，并采用TFrecord标准化方式读取数据，可以作为未来学习和参加比赛的基础。

加载预训练模型与slim的应用参考了该博客：[公输睚信-预训练模型与slim](https://www.jianshu.com/u/c15597fddc5f)
讲解得比较明了。

___

__项目各文件的介绍__

[xuelang_classifier.py](./xuelang_classifier.py)
:分类器模型，定义了一个基于ResNet50的网络结构。

[xuelang_imagePreprocess.py](./xuelang_imagePreprocess.py)
:写了一些图像预处理函数

[xuelang_xmlProcess.py](./xuelang_xmlProcess.py)
:写了一些xml文档处理函数，xml文档存放目标检测信息

[xuelang_generateRecord.py](./xuelang_generateRecord.py)
:生成TFrecord，读取record到slim.dataset

[xuelang_train.py](./xuelang_train.py)
:训练

[xuelang_test.py](./xuelang_test.py)
:测试

[read_model_variables.py](./read_model_variables.py)
:读取存储模型的参数名




