# Quick Start Guide
## Important downloads
- For trained model, go to https://drive.google.com/file/d/1STMPBxQ2d0Kujq2E450fsqFMsz3_X8gi/view?usp=drive_link
- For extracted features, go to https://drive.google.com/file/d/1_UM4LC2_cZ0Dl7EKbaH9wyrG1N1kx4Qg/view?usp=drive_link
  
## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and update with your settings:

```bash
# On Windows
copy .env.example .env

# On Linux/Mac
cp .env.example .env
```

Edit `.env` and set:
- `CHECKPOINT_PATH`: Path to your `epoch_50.pt` file
- `OPENAI_API_KEY`: Your OpenAI API key (optional, for ChatGPT feedback)

### 3. Start the Backend Server

```bash
python start_server.py
```

Or manually:
```bash
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
```

The server will start on `http://localhost:8000`

### 4. Open the Frontend

**Option A: Direct File**
- Open `frontend/index.html` in your web browser

**Option B: HTTP Server (Recommended)**
```bash
cd frontend
python -m http.server 8080
```
Then open `http://localhost:8080` in your browser

## Usage

1. **Upload Audio**: Click "Upload Audio" and select a WAV/MP3 file
2. **Record Audio**: Click "Record Audio" to record from your microphone
3. **View Results**: See the melody comparison graph and AI feedback
4. **Download**: Click "Download Processed Audio" to get the audio with replaced F0

## Features

✅ Upload audio files (WAV, MP3, etc.)
✅ Record audio directly from microphone
✅ Visualize melody comparison (uploaded vs predicted)
✅ Get AI-powered feedback using ChatGPT
✅ Download processed audio with replaced F0
✅ Mobile-friendly responsive design

## Troubleshooting

### Backend won't start
- Make sure `epoch_50.pt` exists or set `CHECKPOINT_PATH` environment variable
- Check that port 8000 is not in use
- Install all dependencies: `pip install -r requirements.txt`

### Frontend can't connect to backend
- Make sure backend is running on port 8000
- Check browser console for CORS errors
- Update `API_URL` in `frontend/index.html` if backend runs on different port

### ChatGPT feedback not working
- Set `OPENAI_API_KEY` in `.env` file
- Get API key from https://platform.openai.com/api-keys
- Feature is optional - app works without it

### F0 replacement quality
- For best results, install `pyworld`: `pip install pyworld`
- Without pyworld, a simpler pitch-shift method is used

## API Endpoints

- `POST /api/process` - Process audio and return graph data + feedback
- `POST /api/replace_f0` - Replace F0 and return processed audio file
- `GET /api/health` - Health check endpoint

## Mobile Usage

The frontend is fully responsive and works on mobile browsers:
- Upload files from your device
- Record using mobile microphone
- View graphs and feedback on mobile screens



