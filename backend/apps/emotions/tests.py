from types import SimpleNamespace
from unittest.mock import patch

import torch
from django.contrib.auth import get_user_model
from django.template.loader import render_to_string
from django.test import SimpleTestCase, TestCase
from django.urls import Resolver404, resolve, reverse

from ml_models import facial_emotion
from ml_models.voice_emotion import VoiceEmotionDetector


class FacialEmotionDetectorTests(SimpleTestCase):
    def test_target_emotion_mapping_is_normalized(self):
        fer_scores = {
            'angry': 0.50,
            'disgust': 0.20,
            'fear': 0.10,
            'happy': 0.15,
            'sad': 0.03,
            'surprise': 0.02,
            'neutral': 0.00,
        }

        result = facial_emotion.narrow_to_target_emotions(fer_scores)

        self.assertEqual(
            set(result.keys()),
            {'happy', 'sad', 'angry', 'surprised', 'scared'},
        )
        self.assertAlmostEqual(sum(result.values()), 1.0)


class FakeBatch(dict):
    def to(self, _device):
        return self


class FakeFeatureExtractor:
    def __init__(self):
        self.call_kwargs = None

    def __call__(self, _audio, **kwargs):
        self.call_kwargs = kwargs
        return FakeBatch(
            input_features=torch.zeros((1, 128, 3000)),
            attention_mask=torch.ones((1, 3000), dtype=torch.long),
        )


class FakeVoiceModel:
    def __init__(self):
        self.input_features = None

    def __call__(self, *, input_features):
        self.input_features = input_features
        return SimpleNamespace(
            logits=torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
        )


class FakeLoadableVoiceModel:
    def __init__(self):
        self.device = None
        self.is_evaluating = False

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.is_evaluating = True
        return self


class VoiceEmotionPreprocessingTests(SimpleTestCase):
    def setUp(self):
        self.detector = VoiceEmotionDetector()

    def _speech_sample(self):
        sample_rate = self.detector.SAMPLE_RATE
        silence = torch.zeros(int(0.4 * sample_rate))
        time = torch.arange(int(1.5 * sample_rate)) / sample_rate
        speech = 0.5 * torch.sin(2 * torch.pi * 220 * time)
        return torch.cat((silence, speech, silence)).unsqueeze(0)

    def test_preprocessing_is_invariant_to_recording_volume(self):
        speech = self._speech_sample()

        quiet, quiet_rate, _ = self.detector._prepare_waveform(
            speech * 0.02,
            self.detector.SAMPLE_RATE,
        )
        loud, loud_rate, _ = self.detector._prepare_waveform(
            speech * 0.8,
            self.detector.SAMPLE_RATE,
        )

        self.assertEqual(quiet_rate, self.detector.SAMPLE_RATE)
        self.assertEqual(loud_rate, self.detector.SAMPLE_RATE)
        torch.testing.assert_close(quiet, loud, rtol=1e-4, atol=1e-5)
        normalized_rms = float(torch.sqrt(torch.mean(quiet.square())))
        self.assertAlmostEqual(normalized_rms, self.detector.TARGET_RMS, places=4)

    def test_silent_audio_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'No audible speech'):
            self.detector._prepare_waveform(
                torch.zeros((1, self.detector.SAMPLE_RATE)),
                self.detector.SAMPLE_RATE,
            )

    def test_whisper_feature_normalization_is_enabled(self):
        feature_extractor = FakeFeatureExtractor()
        model = FakeVoiceModel()
        self.detector.feature_extractor = feature_extractor
        self.detector.model = model

        self.detector._predict_segment(torch.ones((1, self.detector.SAMPLE_RATE)))

        self.assertTrue(feature_extractor.call_kwargs['do_normalize'])
        self.assertFalse(feature_extractor.call_kwargs['return_attention_mask'])
        self.assertTrue(feature_extractor.call_kwargs['truncation'])
        self.assertEqual(
            feature_extractor.call_kwargs['max_length'],
            self.detector.MAX_SEGMENT_SECONDS * self.detector.SAMPLE_RATE,
        )

    def test_attention_mask_is_not_forwarded_to_audio_classifier(self):
        feature_extractor = FakeFeatureExtractor()
        model = FakeVoiceModel()
        self.detector.feature_extractor = feature_extractor
        self.detector.model = model

        self.detector._predict_segment(torch.ones((1, self.detector.SAMPLE_RATE)))

        # FakeFeatureExtractor returns an attention_mask, while FakeVoiceModel's
        # strict signature would raise TypeError if it were forwarded.
        self.assertEqual(model.input_features.shape, (1, 128, 3000))

    def test_model_labels_are_mapped_to_application_labels(self):
        scores = torch.tensor([0.10, 0.05, 0.20, 0.10, 0.15, 0.10, 0.30])
        labels = {
            0: 'angry',
            1: 'disgust',
            2: 'fearful',
            3: 'happy',
            4: 'neutral',
            5: 'sad',
            6: 'surprised',
        }

        result = self.detector._canonicalize_scores(scores, labels)

        self.assertIn('fear', result)
        self.assertIn('surprise', result)
        self.assertNotIn('fearful', result)
        self.assertNotIn('surprised', result)
        self.assertAlmostEqual(result['fear'], 0.20)
        self.assertAlmostEqual(result['surprise'], 0.30)
        self.assertAlmostEqual(sum(result.values()), 1.0)

    @patch('ml_models.voice_emotion.AutoModelForAudioClassification.from_pretrained')
    @patch('ml_models.voice_emotion.AutoFeatureExtractor.from_pretrained')
    def test_model_load_can_retry_after_a_transient_failure(
        self,
        load_feature_extractor,
        load_model,
    ):
        feature_extractor = FakeFeatureExtractor()
        model = FakeLoadableVoiceModel()
        load_feature_extractor.side_effect = [
            OSError('Local cache is incomplete'),
            OSError('Temporary connection failure'),
            feature_extractor,
        ]
        load_model.return_value = model

        self.assertFalse(self.detector._load_model())
        self.assertFalse(self.detector._model_loaded)
        self.assertIsNone(self.detector.model)

        self.assertTrue(self.detector._load_model(force_retry=True))
        self.assertTrue(self.detector._model_loaded)
        self.assertIs(self.detector.feature_extractor, feature_extractor)
        self.assertIs(self.detector.model, model)
        self.assertTrue(model.is_evaluating)
        self.assertTrue(
            load_feature_extractor.call_args_list[0].kwargs['local_files_only']
        )
        self.assertTrue(load_model.call_args.kwargs['local_files_only'])

    def test_model_load_failure_is_not_reported_as_invalid_audio(self):
        self.detector._model_load_error = ImportError(
            'tokenizers version is incompatible with transformers'
        )

        with patch.object(self.detector, '_load_model', return_value=False):
            result = self.detector.detect_from_audio('unused.wav')

        self.assertEqual(result['error_code'], 'voice_model_unavailable')
        self.assertIn('dependencies are incompatible', result['error'])
        self.assertFalse(result['audio_processed'])


class BrowserCaptureControlsTests(SimpleTestCase):
    def setUp(self):
        self.page = render_to_string('emotion_detect.html')

    def test_camera_and_microphone_permission_controls_are_rendered(self):
        self.assertIn('id="cameraStartBtn"', self.page)
        self.assertIn('id="cameraPreview"', self.page)
        self.assertIn('id="microphoneStartBtn"', self.page)
        self.assertIn('id="microphoneStopBtn"', self.page)
        self.assertIn('navigator.mediaDevices.getUserMedia', self.page)

    def test_captured_media_reuses_existing_detection_endpoint(self):
        endpoint = "fetch('/api/emotions/logs/detect/'"
        self.assertEqual(self.page.count(endpoint), 2)
        self.assertIn("formData.append('image', imageFile)", self.page)
        self.assertIn("formData.append('audio', audioFile)", self.page)
        self.assertIn('convertRecordingToWav', self.page)
        self.assertIn("data.error_code !== 'voice_model_unavailable'", self.page)


class NavigationActiveStateTests(TestCase):
    routes = ('home', 'emotion_detect', 'chatbot', 'dashboard')

    @classmethod
    def setUpTestData(cls):
        cls.user = get_user_model().objects.create_user(
            username='navigation-user',
            password='test-password',
        )

    def setUp(self):
        self.client.force_login(self.user)

    def test_only_the_current_page_is_marked_active(self):
        for route_name in self.routes:
            with self.subTest(route=route_name):
                response = self.client.get(reverse(route_name))

                self.assertEqual(response.status_code, 200)
                self.assertContains(response, 'aria-current="page"', count=1)
                self.assertContains(
                    response,
                    f'data-nav-route="{route_name}" aria-current="page"',
                )
                self.assertContains(
                    response,
                    f'href="{reverse(route_name)}"',
                )

    def test_anonymous_users_see_chatbot_but_not_emotion_detection(self):
        self.client.logout()

        response = self.client.get(reverse('home'))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, 'data-nav-route="emotion_detect"')
        self.assertContains(response, 'data-nav-route="chatbot"')
        self.assertContains(response, f'href="{reverse("chatbot")}"')


class DashboardHistoryTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = get_user_model().objects.create_user(
            username='history-user',
            password='test-password',
        )

    def test_authenticated_history_only_shows_recent_emotions(self):
        self.client.force_login(self.user)

        response = self.client.get(reverse('dashboard'))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Review recent emotion checks')
        self.assertContains(response, 'Recent emotions')
        self.assertContains(response, 'id="emotion-logs"')
        self.assertContains(response, "fetchJson('/api/emotions/logs/')")
        self.assertNotContains(response, 'Recent checks')
        self.assertNotContains(response, 'Insights')
        self.assertNotContains(response, 'emotion-insights')
        self.assertNotContains(response, '/api/recommendations/insights/')

    def test_anonymous_history_copy_does_not_mention_insights(self):
        response = self.client.get(reverse('dashboard'))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Review recent emotion checks')
        self.assertNotContains(response, 'insights', status_code=200)

    def test_insights_api_route_is_removed(self):
        with self.assertRaises(Resolver404):
            resolve('/api/recommendations/insights/')
