from django.contrib.auth import views
from django.urls import path, include

urlpatterns = [
    path('newuser/', views.NewUser.as_view(), name='newuser'),
    path('users/', include('django.contrib.auth.urls')),

]
