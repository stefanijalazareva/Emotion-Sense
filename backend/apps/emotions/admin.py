from django.contrib import admin
from .models import EmotionLog, UserSession, UserProfile


@admin.register(EmotionLog)
class EmotionLogAdmin(admin.ModelAdmin):
    list_display = ['user', 'emotion_type', 'confidence', 'source', 'timestamp']
    list_filter = ['emotion_type', 'source', 'timestamp']
    search_fields = ['user__username']
    readonly_fields = ['timestamp']
    date_hierarchy = 'timestamp'


@admin.register(UserSession)
class UserSessionAdmin(admin.ModelAdmin):
    list_display = ['user', 'start_time', 'end_time', 'is_active', 'dominant_emotion', 'total_emotions_detected']
    list_filter = ['is_active', 'dominant_emotion', 'start_time']
    search_fields = ['user__username']
    readonly_fields = ['start_time', 'end_time']


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'enable_face_detection', 'enable_voice_detection', 'total_sessions']
    list_filter = ['enable_face_detection', 'enable_voice_detection']
    search_fields = ['user__username']
