# Melody Transformer Frontend

A web application for analyzing melodies using AI and replacing fundamental frequencies.

## Features

- 🎵 Upload audio files or record directly from microphone
- 📊 Visualize melody comparison (uploaded vs predicted)
- 🤖 Get AI-powered feedback using ChatGPT
- ⬇️ Download processed audio with replaced F0
- 📱 Mobile-friendly responsive design

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create a `.env` file or set environment variables:

```bash
# Path to your checkpoint file
export CHECKPOINT_PATH="epoch_50.pt"

# OpenAI API key for ChatGPT feedback (optional)
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Start the Backend Server

```bash
cd backend
python api.py
```

Or using uvicorn directly:

```bash
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open the Frontend

Open `frontend/index.html` in your web browser, or serve it using a simple HTTP server:

```bash
cd frontend
python -m http.server 8080
```

Then open `http://localhost:8080` in your browser.

## API Endpoints

- `POST /api/process` - Process audio and get graph data + feedback
- `POST /api/replace_f0` - Replace F0 in audio and return processed file
- `GET /api/health` - Health check

## Notes

- The frontend expects the backend to run on `http://localhost:8000`
- For production, update the `API_URL` in `frontend/index.html`
- ChatGPT integration requires an OpenAI API key (optional)
- For high-quality F0 replacement, `pyworld` is recommended but not required

## Mobile Support

The frontend is fully responsive and works on mobile devices. Users can:
- Upload audio files from their device
- Record audio directly using the microphone
- View graphs and feedback on mobile browsers
