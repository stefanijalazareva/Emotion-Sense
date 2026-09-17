from django.contrib import admin
from .models import ChatSession, ChatMessage, ChatbotContext


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ['user', 'start_time', 'end_time', 'is_active', 'initial_emotion', 'total_messages']
    list_filter = ['is_active', 'initial_emotion', 'start_time']
    search_fields = ['user__username']
    readonly_fields = ['start_time', 'end_time']


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ['session', 'sender', 'message_preview', 'detected_emotion', 'timestamp']
    list_filter = ['sender', 'detected_emotion', 'timestamp']
    search_fields = ['message', 'session__user__username']
    readonly_fields = ['timestamp']
    date_hierarchy = 'timestamp'
    
    def message_preview(self, obj):
        return obj.message[:50] + '...' if len(obj.message) > 50 else obj.message
    message_preview.short_description = 'Message'


@admin.register(ChatbotContext)
class ChatbotContextAdmin(admin.ModelAdmin):
    list_display = ['session', 'mood_trend', 'updated_at']
    list_filter = ['mood_trend']
    readonly_fields = ['updated_at']
