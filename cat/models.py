from django.db import models

# Create your models here.
class Catinfo(models.Model):
    name = models.CharField(max_length=10)  # 名字
    nameinfo = models.CharField(max_length=1000)  # 名字信息
    feature = models.CharField(max_length=1000)  # 特征
    livemethod = models.CharField(max_length=1000)  # 生活习惯
    feednn = models.CharField(max_length=1000)  # 饲养需知
    feedmethod = models.CharField(max_length=1000)  # 饲养方法