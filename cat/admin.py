from django.contrib import admin
from .models import *

# 自定义管理页面
class CatinfoAdmin(admin.ModelAdmin):
    list_display = ['id', 'name', 'nameinfo', 'feature', 'livemethod', 'feednn', 'feedmethod']

# Register your models here.
admin.site.register(Catinfo, CatinfoAdmin)