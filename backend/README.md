# EmotionSense Backend

Django backend for the EmotionSense application.

## Setup

Run these commands from the `backend` directory:

```bash
pip install -r ../requirements.txt
cp ../.env.example ../.env
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

## Apps

| App | Purpose |
|-----|---------|
| `emotions` | Facial and voice emotion detection |
| `chatbot` | Groq-powered emotion-aware chatbot |

## API Endpoints

| Prefix | Key routes |
|--------|------------|
| `/api/emotions/` | `logs/detect/`, `sessions/`, `profile/me/` |
| `/api/chatbot/` | `messages/send/`, `sessions/` |

