from rest_framework import serializers

from .models import EmotionLog, UserProfile, UserSession


class EmotionLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = EmotionLog
        fields = ['id', 'emotion_type', 'confidence', 'source', 'timestamp', 'raw_data']
        read_only_fields = ['id', 'timestamp']


class UserSessionSerializer(serializers.ModelSerializer):
    emotions = EmotionLogSerializer(many=True, read_only=True)
    
    class Meta:
        model = UserSession
        fields = ['id', 'start_time', 'end_time', 'is_active', 'dominant_emotion', 
                  'average_confidence', 'total_emotions_detected', 'emotions']
        read_only_fields = ['id', 'start_time', 'end_time', 'dominant_emotion', 
                            'average_confidence', 'total_emotions_detected']


class UserProfileSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source='user.username', read_only=True)
    
    class Meta:
        model = UserProfile
        fields = ['id', 'username', 'preferred_music_genre', 'enable_voice_detection', 
                  'enable_face_detection', 'total_sessions']
        read_only_fields = ['id', 'total_sessions']
