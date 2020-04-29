import os
from PIL import Image
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.layers import Conv2D, MaxPooling2D  # 卷积层 池化层

"""

cattrain就是为了训练训练集，生成权重文件.h5

"""

#--------------------------------------------------------------------------------------------
# 将训练集图片转换成高维数组
ima1 = os.listdir('./cat/train')  # 读取train目录下的所有文件，放入ima1对象中


def read_image1(filename):
    img = Image.open('./cat/train/'+filename).convert('RGB')
    return np.array(img)  # 读取出来的图片转换成一个np的数组并返回


x_train = []  # 声明x_train数组

for i in ima1:  # 读取遍历train文件夹下的图片
    x_train.append(read_image1(i))

x_train = np.array(x_train)  # x_train转换成np类型的数组

# 根据文件名提取标签
y_train = []
for filename in ima1:
    y_train.append(int(filename.split('_')[0]))  # '_'分割线之前的数字读取添加到y_train数组中

y_train = np.array(y_train)  # y_train转换成np类型的数组

# -----------------------------------------------------------------------------------------
# 将测试集图片转换成数组
ima2 = os.listdir('./cat/test')


def read_image2(filename):
    img = Image.open('./cat/test/'+filename).convert('RGB')
    return np.array(img)


x_test = []

for i in ima2:
    x_test.append(read_image2(i))

x_test = np.array(x_test)
# print(x_test)

# 根据文件名提取标签
y_test = []
for filename in ima2:
    y_test.append(int(filename.split('_')[0]))

y_test = np.array(y_test)
# print(y_test)

#-------------------------------------------------------------------------------------
# 将标签转换格式
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# print(y_train, y_test)

# 将特征点从0~255转换成0~1提高特征提取精度，加快收敛（归一化）
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


# 类似 VGG 的卷积神经网络
model = Sequential()  # 宣告一个Network

# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.

# 输入: 3 通道 100x100 像素图像 -> (100, 100, 3) 张量。
# 使用 32 个大小为 3x3 的卷积滤波器。

"""
model.add(Dense(input_dim=28*28,units=500,activation='relu'))
Dense意思就是说你加一个全连接网络，可以加其他的，比如加Con2d，就是加一个convolution layer
"""

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# flatten就是feature map拉直，拉直之后就可以丢到fully connected feedforward netwwork
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))  # 多分类器

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)  # 随机梯度下降
# 第二过程你要做一下configuration，你要定义loss function，选一个optimizer，以及评估指标metrics，
# 其实所有的optimizer都是Gradent descent based，只是有不同的方法来决定learning rate，
# 比如Adam，SGD，RMSprop，Adagrad，Adalta，Adamax ，Nadam等，
# categorical_crossentropy 多分类的对数损失函数
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# batch_size 一次批处理完成的图片数量
model.fit(x_train, y_train, batch_size=10, epochs=32)

model.save_weights('./cat/cat_weights.h5', overwrite=True)

score = model.evaluate(x_test, y_test, batch_size=10)
print(score)
# classes = model.predict_classes(x_test)[0]
#
# test_accuracy = np.mean(np.equal(y_test, classes))
# print("accuarcy:", test_accuracy)



