import logging
import re
from typing import Any, Dict, List, Optional

import httpx
from django.conf import settings
from groq import Groq

logger = logging.getLogger(__name__)


class ChatbotServiceError(RuntimeError):
    """Raised when Groq cannot generate a chatbot response."""


class EmotionAwareChatbot:
    EMOTION_LABELS = ('happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral', 'worried',)
    EMOTION_MARKER = re.compile(
        r'^\s*\[emotion\s*:\s*'
        r'(happy|sad|angry|fear|surprise|disgust|neutral|worried)'
        r'\s*:\s*(0(?:\.\d+)?|1(?:\.0+)?)\s*\]\s*',
        re.IGNORECASE,
    )

    def __init__(self):
        self._client: Optional[Groq] = None
        self._http_client: Optional[httpx.Client] = None
        configured_model = getattr(settings, 'GROQ_MODEL', 'openai/gpt-oss-120b')
        self.model = str(configured_model).strip() or 'openai/gpt-oss-120b'
        configured_key = getattr(settings, 'GROQ_API_KEY', '') or ''
        self.api_key = str(configured_key).strip().strip('"').strip("'")
        self.system_prompt = """You are EmotionSense AI, a compassionate and empathetic mental health support chatbot.

Your role:
- Provide emotional support and active listening
- Be understanding, non-judgmental, and encouraging
- Ask thoughtful follow-up questions to understand user's feelings
- Offer practical coping strategies when appropriate
- Recognize when professional help might be needed
- Adapt your tone based on the user's current emotional state

Important guidelines:
- Never claim to be a replacement for professional therapy
- Be warm and supportive but maintain appropriate boundaries
- If user expresses thoughts of self-harm, strongly encourage professional help
- Keep responses conversational and concise (2-4 sentences usually)
- Use empathetic language and validate their feelings
- Avoid being overly clinical or robotic

Response format:
- Start every response with exactly [emotion:LABEL:CONFIDENCE].
- LABEL must be one of happy, sad, angry, fear, surprise, disgust, neutral, or worried.
- CONFIDENCE must be a number from 0 to 1 representing the emotion expressed in the user's latest message.
- Write the normal supportive reply immediately after the marker.
- Do not mention or explain the marker in the reply.

Remember: You're a supportive companion, not a therapist."""

    @property
    def client(self) -> Optional[Groq]:
        if self._client is None and self.api_key:
            try:
                self._http_client = httpx.Client(timeout=30.0)
                self._client = Groq(api_key=self.api_key, http_client=self._http_client)
                logger.info('Groq AI client initialized with model %s', self.model)
            except Exception:
                logger.exception('Failed to initialize Groq client')
                self._client = None
                if self._http_client is not None:
                    self._http_client.close()
                    self._http_client = None
        return self._client

    def generate_response(
        self,
        user_message: str,
        current_emotion: Optional[str] = None,
        emotion_confidence: Optional[float] = None,
        conversation_history: Optional[List[Dict]] = None,
        context_summary: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise ChatbotServiceError('Groq API key is not configured.')

        if not self.client:
            raise ChatbotServiceError('Groq client initialization failed.')

        try:
            messages = [{'role': 'system', 'content': self.system_prompt}]

            if current_emotion:
                messages.append({
                    'role': 'system',
                    'content': self._build_emotion_context(current_emotion, emotion_confidence),
                })

            if context_summary:
                messages.append({
                    'role': 'system',
                    'content': f'Conversation context: {context_summary}',
                })

            if conversation_history:
                for msg in conversation_history[-10:]:
                    messages.append({
                        'role': 'user' if msg['sender'] == 'user' else 'assistant',
                        'content': msg['message'],
                    })

            messages.append({'role': 'user', 'content': user_message})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=512,
                top_p=0.9,
            )

            raw_response = (response.choices[0].message.content or '').strip()
            bot_response, detected_emotion, detected_confidence = (
                self._extract_emotion_marker(raw_response)
            )
            detected_emotion = detected_emotion or self._normalize_emotion(current_emotion) or 'neutral'
            detected_confidence = (
                detected_confidence
                if detected_confidence is not None
                else self._normalize_confidence(emotion_confidence, default=0.5)
            )
            if not bot_response:
                raise ChatbotServiceError('Groq returned an empty response.')
            response_type = self._classify_response_type(bot_response)
            usage = getattr(response, 'usage', None)

            return {
                'response': bot_response,
                'response_type': response_type,
                'emotion_addressed': detected_emotion,
                'detected_emotion': detected_emotion,
                'emotion_confidence': detected_confidence,
                'tokens_used': getattr(usage, 'total_tokens', None),
            }

        except ChatbotServiceError:
            raise
        except Exception as exc:
            logger.exception('Groq API error using model %s: %s', self.model, exc)
            raise ChatbotServiceError(
                'Groq request failed. Check the server log for details.'
            ) from exc

    def _extract_emotion_marker(self, response: str):
        marker = self.EMOTION_MARKER.match(response or '')
        if not marker:
            return (response or '').strip(), None, None

        emotion = marker.group(1).lower()
        confidence = self._normalize_confidence(marker.group(2), default=0.5)
        return (response[marker.end():].strip(), emotion, confidence)

    def _normalize_emotion(self, emotion: Optional[str]) -> Optional[str]:
        normalized = str(emotion or '').strip().lower()
        return normalized if normalized in self.EMOTION_LABELS else None

    @staticmethod
    def _normalize_confidence(confidence, default=0.5) -> float:
        try:
            return max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            return default

    def _build_emotion_context(self, emotion: str, confidence: Optional[float] = None) -> str:
        confidence_text = ''
        if confidence is not None:
            try:
                conf_value = float(confidence)
                confidence_text = f'with {conf_value * 100:.0f}% confidence'
            except (ValueError, TypeError):
                confidence_text = ''

        emotion_guidance = {
            'sad': 'The user is feeling sad. Be extra gentle and validating. Offer comfort and understanding.',
            'angry': 'The user is feeling angry. Acknowledge their frustration without judgment. Help them process their feelings.',
            'fear': 'The user is feeling fearful or anxious. Be calming and reassuring. Help them feel safe.',
            'happy': 'The user is feeling happy. Share in their joy while maintaining the supportive role.',
            'surprise': 'The user seems surprised. Be curious and help them process unexpected feelings.',
            'disgust': 'The user is feeling disgusted or uncomfortable. Validate their reaction and explore the source.',
            'neutral': 'The user appears emotionally neutral. Be warm and inviting to open conversation.',
            'worried': 'The user is worried. Be reassuring and help them break down their concerns.',
        }

        guidance = emotion_guidance.get(emotion, 'Be empathetic and supportive.')
        return f'Current detected emotion: {emotion} {confidence_text}. {guidance}'

    def _classify_response_type(self, response: str) -> str:
        response_lower = response.lower()

        if '?' in response:
            return 'question'
        if any(word in response_lower for word in ['try', 'practice', 'could', 'might want to']):
            return 'suggestion'
        if any(word in response_lower for word in ['understand', 'hear', 'sounds like', 'seems']):
            return 'validation'
        if any(word in response_lower for word in ['professional', 'therapist', 'counselor', 'doctor']):
            return 'referral'
        if any(word in response_lower for word in ['glad', 'wonderful', 'great', 'amazing']):
            return 'encouragement'
        return 'support'

    def generate_conversation_summary(self, messages: List[Dict]) -> str:
        if not messages or len(messages) < 3:
            return ''

        if not self.client:
            return "Conversation about user's emotional state and concerns."

        try:
            conversation_text = '\n'.join(
                f"{msg['sender']}: {msg['message']}" for msg in messages
            )

            summary_prompt = f"""Summarize this conversation in 2-3 sentences, focusing on:
1. Main topics discussed
2. User's primary concerns or emotions
3. Any progress or insights gained

Conversation:
{conversation_text}

Summary:"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{'role': 'user', 'content': summary_prompt}],
                temperature=0.5,
                max_tokens=512,
            )

            return (response.choices[0].message.content or '').strip()

        except Exception:
            logger.exception('Failed to generate conversation summary')
            return "Conversation about user's emotional state and concerns."

    def analyze_mood_trend(self, emotion_history: List[str]) -> str:
        if len(emotion_history) < 3:
            return 'insufficient_data'

        emotion_scores = {
            'happy': 5,
            'surprise': 4,
            'neutral': 3,
            'worried': 2,
            'sad': 1,
            'angry': 1,
            'fear': 1,
            'disgust': 1,
        }

        mid_point = len(emotion_history) // 2
        first_half_avg = sum(emotion_scores.get(e, 3) for e in emotion_history[:mid_point]) / mid_point
        second_half_avg = (
            sum(emotion_scores.get(e, 3) for e in emotion_history[mid_point:])
            / (len(emotion_history) - mid_point)
        )

        diff = second_half_avg - first_half_avg

        if diff > 0.5:
            return 'improving'
        if diff < -0.5:
            return 'declining'
        return 'stable'


emotion_chatbot = EmotionAwareChatbot()
