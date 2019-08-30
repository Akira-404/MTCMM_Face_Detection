# mtcnn_read_photo
使用mtcnn去截取人脸部分

开发环境win10x64  
使用框架TensorFlow+python3+opencv2  

MTCNN，Multi-task convolutional neural network（多任务卷积神经网络）。分为P-Net、R-Net、和O-Net三层网络结构。  

mtcnn主要使用技术：
1、IoU（交并比）
2、Bounding-Box regression
3、NMS（非极大值抑制）
4、PRelu
实验效果：
![](https://github.com/omega-Lee/mtcnn_read_photo/raw/master/Face/nana.jpg) 
![](https://github.com/omega-Lee/mtcnn_read_photo/blob/master/Face/nana.jpg)
