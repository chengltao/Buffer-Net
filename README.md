# Buffer-Net

A deep-learning framework for detecting genus of bacteria was designed. It utilizes buffer and down sampling layers to extract the spatial-spectral features of hyperspectral microscopic images.   



## The description of each source code:


#### model:
Pretrained weights of the model were saved as *.pth* format. Since it was bigger than 25MB, we uploaded it at https://drive.google.com/drive/my-drive. And we suggested that the weight file should be placed in this directory.

#### BufferNet.py:

The architecture of BufferNet was implemented as a class *bufferNet*. 


#### BufferNet_main.py:

In this file, loading data, training and testing networks were implemented. The function, *plot_confusion_matrix*, could be used to present the confusion matrix of classification.

#### requirements.txt：
The necessary packages were listed.

#### Contact:
Dr. Chenglong Tao: chengltao@126.com

Prof. Dr. Bingliang Hu: hbl@opt.ac.cn

Prof. Dr. Zhoufeng Zhang: zhangzhoufeng@opt.ac.cn

