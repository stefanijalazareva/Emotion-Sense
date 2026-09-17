from django.contrib import admin
from django.contrib.auth import views as auth_views
from django.urls import include, path
from django.views.generic import TemplateView

from .forms import UserLoginForm
from .views import signup_view

urlpatterns = [
    path('admin/', admin.site.urls),

    path(
        'accounts/login/',
        auth_views.LoginView.as_view(
            template_name='registration/login.html',
            authentication_form=UserLoginForm,
            redirect_authenticated_user=True,
        ),
        name='login',
    ),
    path(
        'accounts/logout/',
        auth_views.LogoutView.as_view(),
        name='logout',
    ),
    path('accounts/signup/', signup_view, name='signup'),

    path('api/emotions/', include('apps.emotions.urls')),
    path('api/chatbot/', include('apps.chatbot.urls')),

    path('', TemplateView.as_view(template_name='home.html'), name='home'),
    path('emotion-detect/', TemplateView.as_view(template_name='emotion_detect.html'), name='emotion_detect'),
    path('chatbot/', TemplateView.as_view(template_name='chatbot.html'), name='chatbot'),
    path('dashboard/', TemplateView.as_view(template_name='dashboard.html'), name='dashboard'),
]
