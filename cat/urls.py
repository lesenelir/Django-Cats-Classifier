from django.conf.urls import url
from cat import views

urlpatterns = [
    url(r'^$', views.index),
    url(r'^info/', views.catinfo),
]