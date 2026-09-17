import logging
import threading
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchaudio
from torchaudio.transforms import Resample
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class VoiceEmotionDetector:
    MODEL_ID = 'firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3'
    SAMPLE_RATE = 16_000
    MAX_AUDIO_SECONDS = 120
    MAX_SEGMENT_SECONDS = 30
    MIN_SPEECH_SECONDS = 0.35
    MIN_RMS = 1e-5
    TARGET_RMS = 0.1
    SILENCE_THRESHOLD_RATIO = 0.02
    TRIM_PADDING_SECONDS = 0.2
    MODEL_RETRY_SECONDS = 10

    APP_EMOTIONS = (
        'angry',
        'disgust',
        'fear',
        'happy',
        'neutral',
        'sad',
        'surprise',
    )
    LABEL_ALIASES = {
        'anger': 'angry',
        'angry': 'angry',
        'disgust': 'disgust',
        'disgusted': 'disgust',
        'fear': 'fear',
        'fearful': 'fear',
        'scared': 'fear',
        'happiness': 'happy',
        'happy': 'happy',
        'calm': 'neutral',
        'neutral': 'neutral',
        'sad': 'sad',
        'sadness': 'sad',
        'surprise': 'surprise',
        'surprised': 'surprise',
    }

    def __init__(self):
        self.feature_extractor = None
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._model_loaded = False
        self._model_load_error = None
        self._last_model_load_attempt = 0.0
        self._model_load_lock = threading.Lock()

    def _load_pretrained_components(self):
        try:
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.MODEL_ID,
                local_files_only=True,
            )
            model = AutoModelForAudioClassification.from_pretrained(
                self.MODEL_ID,
                local_files_only=True,
            )
            return feature_extractor, model
        except OSError as cache_error:
            logger.info(
                'Voice model is not fully available in the local cache; trying the model hub: %s',
                cache_error,
            )

        try:
            feature_extractor = AutoFeatureExtractor.from_pretrained(self.MODEL_ID)
            model = AutoModelForAudioClassification.from_pretrained(self.MODEL_ID)
            return feature_extractor, model
        except Exception as download_error:
            raise OSError(
                'The local voice model cache could not be used and the online model '
                f'download failed: {download_error}'
            ) from download_error

    def _load_model(self, force_retry: bool = False) -> bool:
        if self.model is not None and self.feature_extractor is not None:
            self._model_loaded = True
            return True

        seconds_since_attempt = time.monotonic() - self._last_model_load_attempt
        if (
            not force_retry
            and self._model_load_error is not None
            and seconds_since_attempt < self.MODEL_RETRY_SECONDS
        ):
            return False

        with self._model_load_lock:
            if self.model is not None and self.feature_extractor is not None:
                self._model_loaded = True
                return True

            seconds_since_attempt = time.monotonic() - self._last_model_load_attempt
            if (
                not force_retry
                and self._model_load_error is not None
                and seconds_since_attempt < self.MODEL_RETRY_SECONDS
            ):
                return False

            self._last_model_load_attempt = time.monotonic()
            self._model_loaded = False
            self._model_load_error = None
            self.feature_extractor = None
            self.model = None

            try:
                logger.info('Loading voice emotion model: %s', self.MODEL_ID)
                feature_extractor, model = self._load_pretrained_components()
                model.to(self.device)
                model.eval()

                self.feature_extractor = feature_extractor
                self.model = model
                self._model_loaded = True
                logger.info('Voice emotion model loaded successfully')
                return True
            except Exception as exc:
                self._model_load_error = exc
                self.feature_extractor = None
                self.model = None
                logger.exception('Failed to load voice emotion model: %s', exc)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return False

    def _model_load_error_message(self) -> str:
        detail = str(self._model_load_error or '').strip().lower()

        if any(name in detail for name in ('tokenizers', 'transformers', 'torch', 'torchaudio')):
            return (
                'Voice emotion model dependencies are incompatible. '
                'Reinstall the project requirements and restart the backend.'
            )
        if any(text in detail for text in ('out of memory', 'cannot allocate memory')):
            return (
                'There is not enough memory to initialize the voice emotion model. '
                'Close memory-intensive applications and restart the backend.'
            )
        if any(
            text in detail
            for text in ('connection', 'offline', 'cached files', 'local voice model cache')
        ):
            return (
                'The voice emotion model files are unavailable. Connect to the internet '
                'for the first model download, then try again.'
            )

        return (
            'The voice emotion model could not be initialized. '
            'Please retry in a few seconds or restart the backend.'
        )

    def _prepare_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> Tuple[torch.Tensor, int, float]:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 2 or waveform.shape[-1] == 0:
            raise ValueError('The audio file is empty or has an invalid channel layout.')

        waveform = waveform.float()
        if not torch.isfinite(waveform).all():
            waveform = torch.nan_to_num(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != self.SAMPLE_RATE:
            waveform = Resample(sample_rate, self.SAMPLE_RATE)(waveform)
            sample_rate = self.SAMPLE_RATE

        original_duration = waveform.shape[-1] / sample_rate
        if original_duration > self.MAX_AUDIO_SECONDS:
            raise ValueError(
                f'Audio is too long. Please upload at most {self.MAX_AUDIO_SECONDS} seconds.'
            )

        waveform = waveform - waveform.mean(dim=-1, keepdim=True)
        rms = torch.sqrt(torch.mean(waveform.square()))
        if float(rms) < self.MIN_RMS:
            raise ValueError('No audible speech was found in the audio file.')

        peak = float(waveform.abs().max())
        silence_threshold = max(peak * self.SILENCE_THRESHOLD_RATIO, self.MIN_RMS)
        active_samples = torch.nonzero(
            waveform[0].abs() >= silence_threshold,
            as_tuple=False,
        ).flatten()
        if active_samples.numel() == 0:
            raise ValueError('No audible speech was found in the audio file.')

        padding = int(self.TRIM_PADDING_SECONDS * sample_rate)
        start = max(int(active_samples[0]) - padding, 0)
        end = min(int(active_samples[-1]) + padding + 1, waveform.shape[-1])
        waveform = waveform[:, start:end]

        speech_duration = waveform.shape[-1] / sample_rate
        if speech_duration < self.MIN_SPEECH_SECONDS:
            raise ValueError('The recording is too short. Please provide a longer speech sample.')

        speech_rms = torch.sqrt(torch.mean(waveform.square()))
        waveform = waveform * (self.TARGET_RMS / speech_rms.clamp_min(self.MIN_RMS))

        normalized_peak = waveform.abs().max()
        if float(normalized_peak) > 0.99:
            waveform = waveform * (0.99 / normalized_peak)

        return waveform.contiguous(), sample_rate, float(original_duration)

    def _split_into_segments(self, waveform: torch.Tensor) -> List[torch.Tensor]:
        segment_samples = self.MAX_SEGMENT_SECONDS * self.SAMPLE_RATE
        return list(torch.split(waveform, segment_samples, dim=-1))

    def _predict_segment(self, segment: torch.Tensor) -> torch.Tensor:
        inputs = self.feature_extractor(
            segment.squeeze(0).cpu().numpy(),
            sampling_rate=self.SAMPLE_RATE,
            return_tensors='pt',
            padding='max_length',
            max_length=self.MAX_SEGMENT_SECONDS * self.SAMPLE_RATE,
            truncation=True,
            do_normalize=True,
            return_attention_mask=False,
        )

        input_features = inputs['input_features'].to(self.device)
        with torch.no_grad():
            logits = self.model(input_features=input_features).logits

        return torch.softmax(logits, dim=-1)[0].detach().cpu()

    @classmethod
    def _canonicalize_scores(
        cls,
        scores: torch.Tensor,
        emotion_labels: Dict,
    ) -> Dict[str, float]:
        canonical_scores = {emotion: 0.0 for emotion in cls.APP_EMOTIONS}

        for idx, score in enumerate(scores):
            raw_label = emotion_labels.get(idx, emotion_labels.get(str(idx), ''))
            normalized_label = str(raw_label).strip().lower().replace('-', '_')
            canonical_label = cls.LABEL_ALIASES.get(normalized_label)
            if canonical_label:
                canonical_scores[canonical_label] += float(score)

        total = sum(canonical_scores.values())
        if total <= 0:
            canonical_scores['neutral'] = 1.0
            return canonical_scores

        return {
            emotion: score / total
            for emotion, score in canonical_scores.items()
        }

    def detect_from_audio(self, audio_path: str) -> Dict[str, Any]:
        model_is_ready = self._load_model()

        if not model_is_ready:
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'error': self._model_load_error_message(),
                'error_code': 'voice_model_unavailable',
                'audio_processed': False,
            }

        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            waveform, sample_rate, original_duration = self._prepare_waveform(
                waveform,
                sample_rate,
            )
            segments = self._split_into_segments(waveform)

            weighted_scores: Optional[torch.Tensor] = None
            total_samples = 0
            for segment in segments:
                segment_scores = self._predict_segment(segment)
                segment_samples = segment.shape[-1]
                if weighted_scores is None:
                    weighted_scores = torch.zeros_like(segment_scores)
                weighted_scores += segment_scores * segment_samples
                total_samples += segment_samples

            if weighted_scores is None or total_samples == 0:
                raise ValueError('No usable speech segments were found.')

            averaged_scores = weighted_scores / total_samples
            emotion_scores = self._canonicalize_scores(
                averaged_scores,
                self.model.config.id2label,
            )
            emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = float(emotion_scores[emotion])

            return {
                'emotion': emotion,
                'confidence': confidence,
                'all_emotions': emotion_scores,
                'audio_processed': True,
                'duration_seconds': original_duration,
                'speech_duration_seconds': waveform.shape[-1] / sample_rate,
                'segments_analyzed': len(segments),
                'model_used': 'Whisper Large V3 Speech Emotion Recognition',
            }

        except ValueError as exc:
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'error': str(exc),
                'audio_processed': False,
            }
        except RuntimeError as exc:
            if 'Unknown extension' in str(exc):
                error_message = (
                    'Unsupported audio format. Please use WAV, MP3, FLAC, OGG, or M4A.'
                )
            else:
                logger.exception('Voice detection runtime error')
                error_message = f'Error processing audio: {exc}'
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'error': error_message,
                'audio_processed': False,
            }
        except Exception as exc:
            logger.exception('Voice detection error')
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'error': f'Error processing audio: {exc}',
                'audio_processed': False,
            }


voice_detector = VoiceEmotionDetector()
