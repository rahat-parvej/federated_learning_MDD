from django.urls import path
from . import views

urlpatterns = [
    path('', views.evalute_view,name='test'),
    path('test_result', views.evalute_result,name='test_result'),
]