# Tianchi_xuelangAI_ResNet50withSlim

___本项目是天池大数据竞赛-雪浪AI布匹瑕疵检测比赛初赛的代码。___  
___因为参加时间已经临近初赛结尾，所以只是实现了代码功能，没有仔细调参。可以作为经验在以后的比赛中深入学习。___  

一开始在天池网站上看到这个比赛的时候已经接近尾声了，所以只是调通了整体代码，并没有深入研究模型调参与具体方法。这个
比赛是布匹瑕疵检测，基本上就是图像的目标检测任务。但是初赛只要求做到分类就可以，所以只是简单地将模型写成了二分类。整个项目采用
了tensorflow的slim模块加载预训练的ResNetV2-50模型，并采用TFrecord标准化方式读取数据，可以作为未来学习和参加比赛的基础。


