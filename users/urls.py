from django.urls import path

from . import views
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('newuser/', views.NewUser.as_view(), name='newuser'),
    path('users/', include('django.contrib.auth.urls')),
   # path('', include('pages.urls')),
]
