from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response

from .groq_service import ChatbotServiceError, emotion_chatbot
from .models import ChatMessage, ChatSession, ChatbotContext
from .serializers import ChatMessageSerializer, ChatSessionSerializer


class ChatSessionViewSet(viewsets.ModelViewSet):
    serializer_class = ChatSessionSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return ChatSession.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    @action(detail=False, methods=['get'])
    def active(self, request):
        session, _created = ChatSession.objects.get_or_create(
            user=request.user,
            is_active=True,
        )
        serializer = self.get_serializer(session)
        return Response(serializer.data)

    @action(detail=True, methods=['post'])
    def end(self, request, pk=None):
        session = self.get_object()
        session.end_session()
        serializer = self.get_serializer(session)
        return Response(serializer.data)


class ChatMessageViewSet(viewsets.ModelViewSet):
    serializer_class = ChatMessageSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return ChatMessage.objects.filter(session__user=self.request.user)

    @action(detail=False, methods=['get'])
    def history(self, request):
        recent_messages = list(self.get_queryset().order_by('-timestamp')[:200])
        recent_messages.reverse()
        serializer = self.get_serializer(recent_messages, many=True)
        latest_emotional_message = next(
            (
                message
                for message in reversed(recent_messages)
                if message.sender == 'user' and message.detected_emotion
            ),
            None,
        )
        latest_session = (
            ChatSession.objects.filter(user=request.user)
            .order_by('-start_time')
            .first()
        )
        latest_context = None
        if latest_session:
            try:
                latest_context = latest_session.context
            except ChatbotContext.DoesNotExist:
                pass

        return Response(
            {
                'messages': serializer.data,
                'count': len(recent_messages),
                'context': {
                    'current_emotion': (
                        latest_emotional_message.detected_emotion
                        if latest_emotional_message
                        else 'neutral'
                    ),
                    'emotion_confidence': (
                        latest_emotional_message.emotion_confidence
                        if latest_emotional_message
                        else 0.5
                    ),
                    'mood_trend': (
                        latest_context.mood_trend
                        if latest_context and latest_context.mood_trend
                        else 'insufficient_data'
                    ),
                },
            }
        )

    @action(detail=False, methods=['post'], permission_classes=[AllowAny])
    def send(self, request):
        user_message = request.data.get('message')
        provided_emotion = str(request.data.get('emotion') or '').strip().lower()
        if provided_emotion not in emotion_chatbot.EMOTION_LABELS:
            provided_emotion = None
        provided_confidence = emotion_chatbot._normalize_confidence(
            request.data.get('emotion_confidence'),
            default=0.5,
        )

        if not user_message:
            return Response(
                {'error': 'Message is required'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        is_authenticated = request.user.is_authenticated

        if is_authenticated:
            session, created = ChatSession.objects.get_or_create(
                user=request.user,
                is_active=True,
            )

            context, _ = ChatbotContext.objects.get_or_create(session=session)

            conversation_history = list(
                session.messages.values('sender', 'message').order_by('-timestamp')[:20]
            )[::-1]

        else:
            session = None
            context = None
            conversation_history = []

        try:
            ai_result = emotion_chatbot.generate_response(
                user_message=user_message,
                current_emotion=provided_emotion,
                emotion_confidence=provided_confidence,
                conversation_history=conversation_history,
                context_summary=context.conversation_summary if context else None,
            )
        except ChatbotServiceError as exc:
            return Response({'error': str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        detected_emotion = str(
            ai_result.get('detected_emotion') or provided_emotion or 'neutral'
        ).strip().lower()
        if detected_emotion not in emotion_chatbot.EMOTION_LABELS:
            detected_emotion = 'neutral'
        emotion_confidence = emotion_chatbot._normalize_confidence(
            ai_result.get('emotion_confidence'),
            default=provided_confidence if provided_emotion else 0.5,
        )

        if is_authenticated:
            if created and detected_emotion:
                session.initial_emotion = detected_emotion
                session.save(update_fields=['initial_emotion'])

            user_msg = ChatMessage.objects.create(
                session=session,
                sender='user',
                message=user_message,
                detected_emotion=detected_emotion,
                emotion_confidence=emotion_confidence,
            )

            emotion_history = context.emotion_history or []
            emotion_history.append(detected_emotion)
            context.emotion_history = emotion_history[-100:]
            context.mood_trend = emotion_chatbot.analyze_mood_trend(
                context.emotion_history
            )
            context.save()

            bot_msg = ChatMessage.objects.create(
                session=session,
                sender='bot',
                message=ai_result['response'],
                response_type=ai_result.get('response_type', 'support'),
            )

            if session.messages.count() % 5 == 0:
                recent_messages = list(
                    session.messages.values('sender', 'message').order_by('-timestamp')[:10]
                )
                context.conversation_summary = emotion_chatbot.generate_conversation_summary(
                    recent_messages[::-1]
                )
                context.save()

            return Response(
                {
                    'user_message': ChatMessageSerializer(user_msg).data,
                    'bot_message': ChatMessageSerializer(bot_msg).data,
                    'context': {
                        'current_emotion': detected_emotion,
                        'emotion_confidence': emotion_confidence,
                        'mood_trend': context.mood_trend,
                        'message_count': session.messages.count(),
                    },
                },
                status=status.HTTP_201_CREATED,
            )

        guest_emotion_history = request.session.get('chat_emotion_history', [])
        if not isinstance(guest_emotion_history, list):
            guest_emotion_history = []
        guest_emotion_history.append(detected_emotion)
        guest_emotion_history = guest_emotion_history[-100:]
        request.session['chat_emotion_history'] = guest_emotion_history
        guest_mood_trend = emotion_chatbot.analyze_mood_trend(
            guest_emotion_history
        )

        return Response(
            {
                'user_message': {
                    'message': user_message,
                    'sender': 'user',
                    'detected_emotion': detected_emotion,
                    'emotion_confidence': emotion_confidence,
                },
                'bot_message': {
                    'message': ai_result['response'],
                    'sender': 'bot',
                    'response_type': ai_result.get('response_type', 'support'),
                },
                'context': {
                    'current_emotion': detected_emotion,
                    'emotion_confidence': emotion_confidence,
                    'mood_trend': guest_mood_trend,
                    'message_count': len(guest_emotion_history),
                },
                'guest_mode': True,
            },
            status=status.HTTP_200_OK,
        )
