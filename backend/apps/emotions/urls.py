from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import EmotionLogViewSet, UserSessionViewSet, UserProfileViewSet

router = DefaultRouter()
router.register(r'logs', EmotionLogViewSet, basename='emotionlog')
router.register(r'sessions', UserSessionViewSet, basename='usersession')
router.register(r'profile', UserProfileViewSet, basename='userprofile')

urlpatterns = [
    path('', include(router.urls)),
]
