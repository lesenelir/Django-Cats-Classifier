import os
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, MaxPooling2D, Dense, Activation
from keras.optimizers import Adam
from keras.utils import np_utils


# --------------------------------------------------
# 将训练集图片转换成数组
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

# --------------------------------------------------
# 将测试集图片转化成数组
ima2 = os.listdir('./cat/test')


def read_image2(filename):
    img = Image.open('./cat/test/'+filename).convert('RGB')
    return np.array(img)


x_test = []

for i in ima2:
    x_test.append(read_image2(i))

x_test = np.array(x_test)

# 根据文件名提取标签
y_test = []
for filename in ima2:
    y_test.append(int(filename.split('_')[0]))

y_test = np.array(y_test)

# --------------------------------------------------
# 将标签转换格式
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# 将特征点从0~255转换成0~1提高特征提取精度
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0

# setup Neural network CNN
model = Sequential()

# CNN layer = 1 input shape(100, 100, 3)
model.add(Convolution2D(
    input_shape=(100, 100, 3),
    filters=32,  # next layer output (100, 100, 32)
    kernel_size=(5, 5),
    padding='same',
))
model.add(Activation('relu'))
model.add(MaxPooling2D(
    pool_size=(2, 2),  # output next layer (50, 50, 32)
    strides=(2, 2),
    padding='same',
))

# CNN layer = 2
model.add(Convolution2D(
    filters=64,  # next layer output (50, 50, 64)
    kernel_size=(2, 2),
    padding='same',
))
model.add(Activation('relu'))
model.add(MaxPooling2D(
    pool_size=(2, 2),  # output next layer (25, 25, 64)
    strides=(2, 2),
    padding='same',
))

# Fully connected Layer -1
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

# Fully connected Layer -2
model.add(Dense(512))
model.add(Activation('relu'))

# Fully connected Layer -3
model.add(Dense(256))
model.add(Activation('relu'))

# Fully connected Layer -4
model.add(Dense(4))
model.add(Activation('softmax'))

# Define Optimizer
adam = Adam(lr=0.0001)

# Compile the model
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'],
              )

# Fire up the network
model.fit(
    x=x_train,
    y=y_train,
    epochs=32,
    batch_size=15,  # 每次提度更新的样本数
    verbose=1,  # 日志显示模式 进度条
)

# Save work model
model.save('./cat_weights_second.h5')

score = model.evaluate(x_test, y_test, batch_size=15)
print(score)



