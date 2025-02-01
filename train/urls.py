from django.urls import path
from . import views

urlpatterns = [
    path('', views.train_view, name='train'),
    path('training_going', views.training_going, name='training_going'),
]