from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_view, name='predict'),
    path('pred_result', views.pred_result, name='predict_result'),
]