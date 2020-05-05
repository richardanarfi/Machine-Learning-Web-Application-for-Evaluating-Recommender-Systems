"""machinelearning URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
#from pages import views as vw
# from pages import views as vw
from django.conf.urls import url
from django.contrib import admin
from django.urls import path, include
from django.views.generic.base import TemplateView

from . import views
# from pages import views as vw
# from pages import views as vw
from django.conf.urls import url
from django.contrib import admin
from django.urls import path, include
from django.views.generic.base import TemplateView

from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
	path('users/', include('users.urls')),
    path('users/', include('django.contrib.auth.urls')),
	path('dashboard/', TemplateView.as_view(template_name='dashboard.html'), name='dashboard'),
	path('matrix_factorization/', TemplateView.as_view(template_name='MF.html'), name='Matrix_Factorization'),
	path('content_based/', TemplateView.as_view(template_name='CF.html'), name='Content_Based'),
	path('neural_network/', TemplateView.as_view(template_name='NN.html'), name='Neural_Network'),
	path('scratch_pad/', TemplateView.as_view(template_name='SP.html'), name='Scratch_Pad'),
	path('', TemplateView.as_view(template_name='index.html'), name='index'),
	url(r'^NN_model/', views.NN_model),
	url(r'^matrixFactorization', views.matrixFactorization),
]