from django.shortcuts import render
from django.http import HttpResponse
from Django_Cats_Classifier import settings
from cat import models
from cat.prediction import testcat

"""

视图调用模板：
1.找到模板
2.定义上下文
3.渲染模板，把信息传到模版templates
---------------------------------
render函数参数：
第一个参数为request对象
第二个参数为模板文件路径
第三个参数为字典，表示向模板中传递的上下文数据

"""


# Create your views here.
def index(request):
    return render(request, 'index.html')


# 调用模型对图片进行分类
def catinfo(request):
    if request.method == "POST":
        f1 = request.FILES['pic1']

        # 图片上传到pic文件夹
        fname = '%s/pic/%s' % (settings.MEDIA_ROOT, f1.name)  # 将文件写入pic文件夹下
        with open(fname, 'wb') as pic:
            for c in f1.chunks():
                pic.write(c)

        # 接收表单保存图片----传给info信息页面显示
        fname1 = './static/img/download/%s' % f1.name  # 图片存入静态文件img文件夹内
        with open(fname1, 'wb') as pic:
            for c in f1.chunks():
                pic.write(c)

        num = testcat(f1.name)
        if (num == 0):
            num = 4
        # 通过id获取猫的信息
        # 图片上传后根据model中的id查询，调用卷积神经网络图片，预测结果记录在kind中
        kind = models.Catinfo.objects.get(id=num)
        return render(request, 'info.html',
                      {'nameinfo': kind.nameinfo, 'feature': kind.feature, 'livemethod': kind.livemethod,
                       'feednn': kind.feednn, 'feedmethod': kind.feedmethod, 'picname': f1.name})
    else:
        return HttpResponse("上传失败！")
