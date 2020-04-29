# Django-Cats-Classifier
## 项目简介
- **该项目是基于 Keras 的猫种类识别。**
- **找四种类别的猫构建成training data & testing data。搭建CNN网络来训练得到一个好的model。**
- **最后用 Django 框架来做展示页面**

## 项目实现
### 所用技术
- Python 文件操作
- Keras  搭建卷积神经网络
- Django 框架



### CNN
- [Keras中文文档](https://keras.io/zh/getting-started/sequential-model-guide/)


**此次项目训练用到的模型是Keras中文文档当中的类VGG卷积神经网络训练模型：**

```
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=10)
model.save_weights('./cat/cat_weights.h5', overwrite=True)

score = model.evaluate(x_test, y_test, batch_size=32)
```

##### 注意：
>类VGG模型是卷积两次再做一次subsampling即此处的maxpooling，随后再以概率p随机丢弃神经元
>
>`model.add(Dense(4, activation='softmax'))`softmax时，如此处是二分类问题，把4改为2;同理对于一个手写数字识别的模型，改为10。
>
>`model.save_weights('./cat/cat_weights.h5', overwrite=True)`保存权重文件，下次训练的时候直接加载权重文件即可实现识别。


## 项目展示

### 前台展示
- **index**
![index](https://github.com/lesenelir/Django-Cats-Classifier/blob/master/README_Images/1.png?raw=true)

- **info**
![info](https://raw.githubusercontent.com/lesenelir/Django-Cats-Classifier/master/README_Images/2.png)


## 具体步骤
### 1.预处理数据集
```
# 统一图片名字
def ranamesJPG(filepath, kind):
    images = os.listdir(filepath)  
    for name in images:
        os.rename(filepath+name, filepath+kind+'_'+name.split('.')[0]+'.jpg') # 名字替换
```
```
# 统一图片大小并保存
def convertjpg(jpgfile,outdir,width=100,height=100):
    img = Image.open('/2020-Django-Post/Cat_Kinds_Pictures/布偶猫/'+jpgfile)
    try:
        new_img = img.resize((width, height), Image.BILINEAR)
        new_img.save(os.path.join(outdir, os.path.basename(jpgfile)))
    except Exception as e:
        print(e)


for jpgfile in os.listdir('/2020-Django-Post/Cat_Kinds_Pictures/布偶猫/'):
    print(jpgfile)
    convertjpg(jpgfile, "./buou")
```

### 2.创建model
**参考类VGG模型搭建**


### 3.创建Django项目
#### （3.1）创建项目名
`django-admin startproject 项目名称`
#### （3.2）创建应用名
`python manage.py startapp 应用名称`
#### （3.3）设计模型类
```
class Catinfo(models.Model):
    name = models.CharField(max_length=10)  
    nameinfo = models.CharField(max_length=1000)  
    feature = models.CharField(max_length=1000)  
    livemethod = models.CharField(max_length=1000)  
    feednn = models.CharField(max_length=1000) 
    feedmethod = models.CharField(max_length=1000) 
```
#### （3.4）生成迁移文件
`python manage.py makemigrations`
#### （3.5）执行迁移文件
`python manage.py migrate`



## 项目总结
- **项目归属于图像识别类，主要解决的问题是图像当中的多分类问题，选择合适的model是解决此类问题的关键。选择合适的model之前，任何识别项目所得到的training data都可能是不规范的，所以我们需要在选择model之前，对training data 和 testing data 作数据预处理**

- **此次选用keras文档当中的类VGG模型，其实用自己搭建的CNN模型也能得到很好的识别效果。例如training_second.py代码：**

```
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
    batch_size=15,  
    verbose=1,  
)

# Save work model
model.save('./cat_weights_second.h5')

score = model.evaluate(x_test, y_test, batch_size=15)
print(score)
```






