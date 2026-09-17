from django.contrib.auth import login
from django.shortcuts import redirect, render

from .forms import UserRegistrationForm


def signup_view(request):
    if request.user.is_authenticated:
        return redirect('chatbot')

    form = UserRegistrationForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        user = form.save()
        login(request, user)
        return redirect('chatbot')

    return render(request, 'registration/signup.html', {'form': form})

