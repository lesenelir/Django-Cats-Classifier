import os
from PIL import Image
import numpy as np
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.optimizers import SGD
from keras.layers import Conv2D, MaxPooling2D  # 卷积层 池化层


# 将图片保存成100 * 100 的格式
def prepicture(picname):
    img = Image.open('./media/pic/' + picname)
    new_img = img.resize((100, 100), Image.BILINEAR)
    new_img.save(os.path.join('./media/pic/', os.path.basename(picname)))


# 将图片转换成数组类型
def read_image2(filename):
    img = Image.open('./media/pic/' + filename).convert('RGB')
    return np.array(img)


# 对模型进行预测
def testcat(picname):
    # 预处理图片 变成100 x 100
    prepicture(picname)
    x_test = []

    x_test.append(read_image2(picname))

    x_test = np.array(x_test)

    x_test = x_test.astype('float32')
    x_test /= 255

    keras.backend.clear_session()
    model = Sequential()
    # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # flatten
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.load_weights('./cat/cat_weights.h5')  # 加载权重文件
    classes = model.predict_classes(x_test)[0]  # 预测图片
    # target = ['布偶猫', '孟买猫', '暹罗猫', '英短猫']
    # print(target[classes])
    return classes

# print(testcat('cat1.jpg'))
