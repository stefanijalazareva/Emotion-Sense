import os
import tempfile

from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response

from ml_models.facial_emotion import facial_detector
from ml_models.voice_emotion import voice_detector

from .models import EmotionLog, UserProfile, UserSession
from .serializers import EmotionLogSerializer, UserProfileSerializer, UserSessionSerializer

ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}


class EmotionLogViewSet(viewsets.ModelViewSet):
    serializer_class = EmotionLogSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return EmotionLog.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    @action(detail=False, methods=['post'], permission_classes=[AllowAny])
    def detect(self, request):
        source = request.data.get('source', 'face')

        if source == 'face':
            return self._detect_facial_emotion(request)
        if source == 'voice':
            return self._detect_voice_emotion(request)

        return Response(
            {'error': 'Invalid source. Use "face" or "voice"'},
            status=status.HTTP_400_BAD_REQUEST,
        )

    def _get_active_session(self, request):
        session_id = request.data.get('session_id')
        if not session_id:
            return None

        try:
            return UserSession.objects.get(
                id=session_id,
                user=request.user,
                is_active=True,
            )
        except UserSession.DoesNotExist:
            return None

    def _create_emotion_log(self, request, result, source, raw_data):
        if not request.user.is_authenticated:
            return None, None

        session = self._get_active_session(request)
        emotion_log = EmotionLog.objects.create(
            user=request.user,
            emotion_type=result['emotion'],
            confidence=float(result['confidence']),
            source=source,
            session=session,
            raw_data=raw_data,
        )
        return emotion_log, session

    def _build_detection_response(self, result, emotion_log, session, extra_fields=None):
        response_data = {
            'id': emotion_log.id if emotion_log else None,
            'emotion': result['emotion'],
            'confidence': float(result['confidence']),
            'timestamp': emotion_log.timestamp if emotion_log else None,
            'session_id': session.id if session else None,
        }
        if extra_fields:
            response_data.update(extra_fields)
        return response_data

    def _detect_facial_emotion(self, request):
        if 'image' not in request.FILES:
            return Response(
                {'error': 'No image file provided. Please upload an image.'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        image_file = request.FILES['image']
        file_extension = image_file.name.rsplit('.', 1)[-1].lower()

        if file_extension not in ALLOWED_IMAGE_EXTENSIONS:
            return Response(
                {'error': f'Invalid file type. Allowed: {", ".join(sorted(ALLOWED_IMAGE_EXTENSIONS))}'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        tmp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
                for chunk in image_file.chunks():
                    tmp_file.write(chunk)
                tmp_file_path = tmp_file.name

            result = facial_detector.detect_from_image(tmp_file_path)

            if not result.get('face_detected', False):
                return Response(
                    {
                        'error': result.get('error', 'No face detected in image'),
                        'emotion': 'neutral',
                        'confidence': 0.0,
                        'face_detected': False,
                    },
                    status=status.HTTP_200_OK,
                )

            all_emotions_serializable = {
                emotion: float(score)
                for emotion, score in result.get('all_emotions', {}).items()
            }

            emotion_log, session = self._create_emotion_log(
                request,
                result,
                source='face',
                raw_data=all_emotions_serializable,
            )

            return Response(
                self._build_detection_response(
                    result,
                    emotion_log,
                    session,
                    extra_fields={
                        'face_detected': True,
                        'all_emotions': all_emotions_serializable,
                    },
                ),
                status=status.HTTP_201_CREATED,
            )

        except Exception as exc:
            return Response(
                {'error': f'Error processing image: {exc}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        finally:
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

    def _detect_voice_emotion(self, request):
        if 'audio' not in request.FILES:
            return Response(
                {'error': 'No audio file provided. Please upload an audio file.'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        audio_file = request.FILES['audio']
        file_extension = audio_file.name.rsplit('.', 1)[-1].lower()

        if file_extension not in ALLOWED_AUDIO_EXTENSIONS:
            return Response(
                {'error': f'Invalid file type. Allowed: {", ".join(sorted(ALLOWED_AUDIO_EXTENSIONS))}'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        tmp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
                for chunk in audio_file.chunks():
                    tmp_file.write(chunk)
                tmp_file_path = tmp_file.name

            result = voice_detector.detect_from_audio(tmp_file_path)

            if not result.get('audio_processed', False):
                return Response(
                    {
                        'error': result.get('error', 'Error processing audio file'),
                        'error_code': result.get('error_code'),
                        'emotion': 'neutral',
                        'confidence': 0.0,
                        'audio_processed': False,
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            emotion_log, session = self._create_emotion_log(
                request,
                result,
                source='voice',
                raw_data={
                    'duration_seconds': result.get('duration_seconds', 0),
                    'speech_duration_seconds': result.get('speech_duration_seconds', 0),
                    'segments_analyzed': result.get('segments_analyzed', 0),
                    'model_used': result.get('model_used', ''),
                    'all_emotions': result.get('all_emotions', {}),
                },
            )

            return Response(
                self._build_detection_response(
                    result,
                    emotion_log,
                    session,
                    extra_fields={
                        'audio_processed': True,
                        'duration_seconds': result.get('duration_seconds', 0),
                        'speech_duration_seconds': result.get('speech_duration_seconds', 0),
                        'segments_analyzed': result.get('segments_analyzed', 0),
                        'model_used': result.get('model_used', 'Whisper + Emotion Recognition'),
                        'all_emotions': result.get('all_emotions', {}),
                    },
                ),
                status=status.HTTP_201_CREATED,
            )

        except Exception as exc:
            return Response(
                {'error': f'Error processing audio: {exc}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        finally:
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)


class UserSessionViewSet(viewsets.ModelViewSet):
    serializer_class = UserSessionSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return UserSession.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    @action(detail=True, methods=['post'])
    def end(self, request, pk=None):
        session = self.get_object()
        session.end_session()
        serializer = self.get_serializer(session)
        return Response(serializer.data)

    @action(detail=False, methods=['get'])
    def active(self, request):
        session, _created = UserSession.objects.get_or_create(
            user=request.user,
            is_active=True,
        )
        serializer = self.get_serializer(session)
        return Response(serializer.data)


class UserProfileViewSet(viewsets.ModelViewSet):
    serializer_class = UserProfileSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return UserProfile.objects.filter(user=self.request.user)

    @action(detail=False, methods=['get'])
    def me(self, request):
        profile, _created = UserProfile.objects.get_or_create(user=request.user)
        serializer = self.get_serializer(profile)
        return Response(serializer.data)
