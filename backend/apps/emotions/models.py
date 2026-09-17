from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone


class EmotionLog(models.Model):    
    EMOTION_CHOICES = [
        ('happy', 'Happy'),
        ('sad', 'Sad'),
        ('angry', 'Angry'),
        ('fear', 'Fear'),
        ('surprise', 'Surprise'),
        ('disgust', 'Disgust'),
        ('neutral', 'Neutral'),
        ('worried', 'Worried'),
    ]
    
    SOURCE_CHOICES = [
        ('face', 'Facial Recognition'),
        ('voice', 'Voice Analysis'),
        ('combined', 'Face + Voice'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='emotion_logs')
    emotion_type = models.CharField(max_length=20, choices=EMOTION_CHOICES)
    confidence = models.FloatField(help_text="Confidence score (0-1)")
    source = models.CharField(max_length=10, choices=SOURCE_CHOICES)
    timestamp = models.DateTimeField(default=timezone.now)
    session = models.ForeignKey('UserSession', on_delete=models.CASCADE, related_name='emotions', null=True, blank=True)
    
    raw_data = models.JSONField(null=True, blank=True, help_text="Raw emotion detection data")
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['user', '-timestamp']),
            models.Index(fields=['emotion_type']),
        ]
    
    def __str__(self):
        return f"{self.user.username} - {self.emotion_type} ({self.confidence:.2f}) at {self.timestamp}"


class UserSession(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='sessions')
    start_time = models.DateTimeField(default=timezone.now)
    end_time = models.DateTimeField(null=True, blank=True)
    is_active = models.BooleanField(default=True)
    
    dominant_emotion = models.CharField(max_length=20, null=True, blank=True)
    average_confidence = models.FloatField(null=True, blank=True)
    total_emotions_detected = models.IntegerField(default=0)
    
    class Meta:
        ordering = ['-start_time']
    
    def __str__(self):
        return f"{self.user.username} - Session {self.id} ({self.start_time})"
    
    def end_session(self):
        self.end_time = timezone.now()
        self.is_active = False
        
        emotions = self.emotions.all()
        if emotions.exists():
            self.total_emotions_detected = emotions.count()
            self.average_confidence = emotions.aggregate(models.Avg('confidence'))['confidence__avg']
            
            from django.db.models import Count
            dominant = emotions.values('emotion_type').annotate(
                count=Count('emotion_type')
            ).order_by('-count').first()
            
            if dominant:
                self.dominant_emotion = dominant['emotion_type']
        
        self.save()


class UserProfile(models.Model):    
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    
    preferred_music_genre = models.CharField(max_length=50, blank=True)
    enable_voice_detection = models.BooleanField(default=True)
    enable_face_detection = models.BooleanField(default=True)
    
    total_sessions = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'User Profile'
        verbose_name_plural = 'User Profiles'
    
    def __str__(self):
        return f"{self.user.username}'s Profile"
