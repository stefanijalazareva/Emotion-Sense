import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.template.loader import render_to_string
from django.test import SimpleTestCase, TestCase, override_settings

from config import settings as settings_module

from .groq_service import ChatbotServiceError, EmotionAwareChatbot
from .models import ChatMessage, ChatSession, ChatbotContext


class EnvLoadingTests(SimpleTestCase):
    def test_load_environment_files_prioritizes_project_root_env(self):
        with TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir) / 'backend'
            base_dir.mkdir(parents=True, exist_ok=True)
            project_root = Path(tmp_dir)

            (base_dir / '.env').write_text('GROQ_API_KEY=backend-key\n', encoding='utf-8')
            (project_root / '.env').write_text('GROQ_API_KEY=root-key\nOTHER_VALUE=project\n', encoding='utf-8')

            os.environ.pop('GROQ_API_KEY', None)
            os.environ.pop('OTHER_VALUE', None)

            try:
                settings_module._load_environment_files(base_dir, project_root)
                self.assertEqual(os.environ['GROQ_API_KEY'], 'root-key')
                self.assertEqual(os.environ['OTHER_VALUE'], 'project')
            finally:
                os.environ.pop('GROQ_API_KEY', None)
                os.environ.pop('OTHER_VALUE', None)


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        current_message = kwargs['messages'][-1]['content']
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=f'Helpful response about: {current_message}')
                )
            ],
            usage=SimpleNamespace(total_tokens=20),
        )


class ChatbotServiceTests(SimpleTestCase):
    @override_settings(
        GROQ_API_KEY='test-key',
        GROQ_MODEL='openai/gpt-oss-120b',
    )
    def test_different_messages_are_sent_to_groq_and_return_distinct_responses(self):
        chatbot = EmotionAwareChatbot()
        completions = FakeCompletions()
        chatbot._client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions),
        )

        first = chatbot.generate_response('I broke my leg')
        second = chatbot.generate_response('Can you give me tips to feel better?')

        self.assertNotEqual(first['response'], second['response'])
        self.assertEqual(
            [call['messages'][-1]['content'] for call in completions.calls],
            ['I broke my leg', 'Can you give me tips to feel better?'],
        )
        self.assertTrue(
            all(call['model'] == 'openai/gpt-oss-120b' for call in completions.calls)
        )

    @override_settings(GROQ_API_KEY='')
    def test_missing_api_key_is_reported(self):
        chatbot = EmotionAwareChatbot()

        with self.assertRaisesRegex(ChatbotServiceError, 'API key is not configured'):
            chatbot.generate_response('I broke my leg')

    @override_settings(GROQ_API_KEY='')
    def test_missing_api_key_does_not_generate_local_response(self):
        chatbot = EmotionAwareChatbot()

        with self.assertRaises(ChatbotServiceError):
            chatbot.generate_response('I feel so sad, lonely, and hopeless today')

    @override_settings(GROQ_API_KEY='test-key')
    def test_ai_emotion_marker_is_removed_from_the_visible_reply(self):
        chatbot = EmotionAwareChatbot()
        completions = FakeCompletions()
        chatbot._client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions),
        )
        completions.create = lambda **_kwargs: SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='[emotion:worried:0.87] That sounds stressful. How can I help?'
                    )
                )
            ],
            usage=SimpleNamespace(total_tokens=20),
        )

        result = chatbot.generate_response('Everything feels uncertain')

        self.assertEqual(result['detected_emotion'], 'worried')
        self.assertEqual(result['emotion_confidence'], 0.87)
        self.assertEqual(result['response'], 'That sounds stressful. How can I help?')


class ChatbotTemplateTests(SimpleTestCase):
    def test_chat_updates_emotion_and_renders_uploaded_image_as_user_message(self):
        page = render_to_string('chatbot.html')

        self.assertIn('data.context.current_emotion', page)
        self.assertIn("currentEmotionSource = 'chat'", page)
        self.assertIn("addImageMessage('user', file)", page)
        self.assertIn("messageEl.dataset.messageType = 'image'", page)
        self.assertIn('URL.createObjectURL(file)', page)

    def test_chat_supports_camera_capture_for_emotion_images(self):
        page = render_to_string('chatbot.html')

        self.assertIn('id="chatCameraStartBtn"', page)
        self.assertIn('id="chatCameraPreview"', page)
        self.assertIn('id="chatCameraCanvas"', page)
        self.assertIn('navigator.mediaDevices.getUserMedia', page)
        self.assertIn('captureChatCameraPhoto()', page)
        self.assertIn('canvas.toBlob', page)
        self.assertIn('new File(', page)
        self.assertIn('chatCameraRequestId += 1', page)
        self.assertIn('requestId !== chatCameraRequestId', page)
        self.assertIn("track.addEventListener('ended'", page)
        self.assertIn('const context = canvas.getContext', page)
        self.assertIn('captureButton.focus()', page)
        self.assertIn('await processEmotionFile(file)', page)
        self.assertIn("formData.append('image', file)", page)

    def test_chat_labels_the_current_mood_without_a_trend(self):
        page = render_to_string('chatbot.html')

        self.assertIn('Current mood:', page)
        self.assertNotIn('Current cue:', page)
        self.assertNotIn('Trend:', page)
        self.assertNotIn('Tracking...', page)
        self.assertNotIn('id="moodTrend"', page)


class UserAuthenticationTests(TestCase):
    def test_regular_non_staff_user_can_login_and_logout(self):
        user = get_user_model().objects.create_user(
            username='regular-user',
            password='SafePassword123!',
        )

        login_response = self.client.post(
            '/accounts/login/',
            {
                'username': 'regular-user',
                'password': 'SafePassword123!',
            },
        )

        self.assertFalse(user.is_staff)
        self.assertRedirects(login_response, '/chatbot/', fetch_redirect_response=False)
        self.assertEqual(str(user.pk), self.client.session.get('_auth_user_id'))

        logout_response = self.client.post('/accounts/logout/')

        self.assertRedirects(logout_response, '/', fetch_redirect_response=False)
        self.assertNotIn('_auth_user_id', self.client.session)

    def test_signup_creates_and_logs_in_regular_user(self):
        response = self.client.post(
            '/accounts/signup/',
            {
                'username': 'new-user',
                'password1': 'SafePassword123!',
                'password2': 'SafePassword123!',
            },
        )

        user = get_user_model().objects.get(username='new-user')
        self.assertFalse(user.is_staff)
        self.assertRedirects(response, '/chatbot/', fetch_redirect_response=False)
        self.assertEqual(str(user.pk), self.client.session.get('_auth_user_id'))


class ChatMessageViewTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username='chat-user',
            password='SafePassword123!',
        )
        self.client.force_login(self.user)

    @patch('apps.chatbot.views.emotion_chatbot.generate_response')
    def test_groq_failure_returns_service_unavailable(self, generate_response):
        generate_response.side_effect = ChatbotServiceError('Groq API key is not configured.')

        response = self.client.post(
            '/api/chatbot/messages/send/',
            {'message': 'Hello'},
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {'error': 'Groq API key is not configured.'})

    @patch('apps.chatbot.views.emotion_chatbot.generate_response')
    def test_current_message_is_not_duplicated_in_conversation_history(self, generate_response):
        generate_response.return_value = {
            'response': 'A unique response',
            'response_type': 'support',
        }

        first_response = self.client.post(
            '/api/chatbot/messages/send/',
            {'message': 'First message'},
        )
        second_response = self.client.post(
            '/api/chatbot/messages/send/',
            {'message': 'Second message'},
        )

        self.assertEqual(first_response.status_code, 201)
        self.assertEqual(second_response.status_code, 201)
        latest_call = generate_response.call_args.kwargs
        self.assertEqual(latest_call['user_message'], 'Second message')
        self.assertEqual(
            latest_call['conversation_history'],
            [
                {'sender': 'user', 'message': 'First message'},
                {'sender': 'bot', 'message': 'A unique response'},
            ],
        )

    @patch('apps.chatbot.views.emotion_chatbot.generate_response')
    def test_history_survives_logout_and_login_and_is_scoped_to_user(self, generate_response):
        generate_response.return_value = {
            'response': 'Your saved response',
            'response_type': 'support',
        }
        send_response = self.client.post(
            '/api/chatbot/messages/send/',
            {'message': 'Remember this conversation'},
        )
        self.assertEqual(send_response.status_code, 201)

        other_user = get_user_model().objects.create_user(
            username='other-user',
            password='SafePassword123!',
        )
        other_session = ChatSession.objects.create(user=other_user)
        ChatMessage.objects.create(
            session=other_session,
            sender='user',
            message='Private message from another user',
        )

        self.client.post('/accounts/logout/')
        login_response = self.client.post(
            '/accounts/login/',
            {
                'username': 'chat-user',
                'password': 'SafePassword123!',
            },
        )
        self.assertRedirects(login_response, '/chatbot/', fetch_redirect_response=False)

        history_response = self.client.get('/api/chatbot/messages/history/')
        self.assertEqual(history_response.status_code, 200)
        saved_messages = [item['message'] for item in history_response.json()['messages']]
        self.assertEqual(
            saved_messages,
            ['Remember this conversation', 'Your saved response'],
        )
        self.assertNotIn('Private message from another user', saved_messages)

    @patch('apps.chatbot.views.emotion_chatbot.generate_response')
    def test_text_emotion_updates_response_history_and_mood_context(self, generate_response):
        generate_response.return_value = {
            'response': 'I hear how lonely this feels.',
            'response_type': 'validation',
            'detected_emotion': 'sad',
            'emotion_confidence': 0.91,
        }

        response = self.client.post(
            '/api/chatbot/messages/send/',
            {'message': 'I feel very lonely today'},
        )

        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertEqual(data['context']['current_emotion'], 'sad')
        self.assertEqual(data['context']['emotion_confidence'], 0.91)
        self.assertEqual(data['context']['mood_trend'], 'insufficient_data')

        user_message = ChatMessage.objects.get(sender='user')
        self.assertEqual(user_message.detected_emotion, 'sad')
        self.assertEqual(user_message.emotion_confidence, 0.91)
        context = ChatbotContext.objects.get(session=user_message.session)
        self.assertEqual(context.emotion_history, ['sad'])

        history = self.client.get('/api/chatbot/messages/history/').json()
        self.assertEqual(history['context']['current_emotion'], 'sad')
        self.assertEqual(history['context']['emotion_confidence'], 0.91)

    @patch('apps.chatbot.views.emotion_chatbot.generate_response')
    def test_guest_chat_tracks_emotion_and_mood_in_the_session(self, generate_response):
        self.client.logout()
        generate_response.side_effect = [
            {
                'response': 'Sad response',
                'detected_emotion': 'sad',
                'emotion_confidence': 0.8,
            },
            {
                'response': 'Still sad response',
                'detected_emotion': 'sad',
                'emotion_confidence': 0.8,
            },
            {
                'response': 'Happy response',
                'detected_emotion': 'happy',
                'emotion_confidence': 0.9,
            },
        ]

        self.client.post('/api/chatbot/messages/send/', {'message': 'First'})
        self.client.post('/api/chatbot/messages/send/', {'message': 'Second'})
        response = self.client.post(
            '/api/chatbot/messages/send/',
            {'message': 'I am feeling much better'},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['context']['current_emotion'], 'happy')
        self.assertEqual(data['context']['mood_trend'], 'improving')
        self.assertEqual(
            self.client.session['chat_emotion_history'],
            ['sad', 'sad', 'happy'],
        )
