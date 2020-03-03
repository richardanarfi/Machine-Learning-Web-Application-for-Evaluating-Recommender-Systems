from django.shortcuts import render
from django.contrib.auth.forms import UserCreationForm
from django.urls import reverse_lazy
from django.views import generic


class NewUser(generic.CreateView):
    form_class = UserCreationForm
    success_url = reverse_lazy('dashboard')
    template_name = 'newuser.html'