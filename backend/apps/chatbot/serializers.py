from rest_framework import serializers
from .models import ChatSession, ChatMessage, ChatbotContext


class ChatMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatMessage
        fields = ['id', 'sender', 'message', 'timestamp', 'detected_emotion', 
                  'emotion_confidence', 'response_type']
        read_only_fields = ['id', 'timestamp']


class ChatSessionSerializer(serializers.ModelSerializer):
    messages = ChatMessageSerializer(many=True, read_only=True)
    
    class Meta:
        model = ChatSession
        fields = ['id', 'start_time', 'end_time', 'is_active', 'initial_emotion', 
                  'total_messages', 'messages']
        read_only_fields = ['id', 'start_time', 'end_time', 'total_messages']


class ChatbotContextSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatbotContext
        fields = ['id', 'conversation_summary', 'user_concerns', 'suggested_topics', 
                  'emotion_history', 'mood_trend', 'updated_at']
        read_only_fields = ['id', 'updated_at']
