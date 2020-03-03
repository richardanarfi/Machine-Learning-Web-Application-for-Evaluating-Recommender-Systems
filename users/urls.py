from django.urls import path

from . import views

urlpatterns = [
    path('newuser/', views.NewUser.as_view(), name='newuser'),
]