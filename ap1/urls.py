from django.urls import path
from . import views
from .views import *

urlpatterns = [
    path('', views.home, name='home'),
    path('bw_image/<int:pk>/', views.bw_image, name='bw_image'),
    path('chat/', views.chat, name='chat'),
     path('result/<int:scan_id>', results, name="results")
]
