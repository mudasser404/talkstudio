# F5-TTS Voice Cloning API Server

Standalone FastAPI server for F5-TTS voice cloning model. Deploy this on a separate GPU server.

## 📁 Project Structure

```
f5tts_api_server/
├── main.py                 # FastAPI server (main application)
├── requirements.txt        # Python dependencies
├── setup.bat              # Windows setup script
├── start_server.bat       # Windows start script
├── start_server.sh        # Linux/Mac start script
├── download_model.bat     # Download models script
├── Dockerfile             # Docker image with CUDA
├── docker-compose.yml     # Docker compose config
├── .env.example           # Environment variables example
├── .gitignore             # Git ignore rules
└── models/
    ├── download_models.py # Model download script
    └── README.md          # Models documentation
```

## 🖥️ System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10+ | 3.10 |
| GPU | NVIDIA GTX 1080 | RTX 3090 / A100 |
| VRAM | 6GB | 12GB+ |
| RAM | 8GB | 16GB+ |
| Disk | 5GB | 10GB |
| CUDA | 11.8+ | 12.1 |

## 🚀 Quick Start

### Option 1: Windows Setup

```bash
# Step 1: Run setup (creates venv, installs PyTorch & dependencies)
setup.bat

# Step 2: Download models (~1.5GB)
download_model.bat

# Step 3: Start server
start_server.bat
```

### Option 2: Linux/Mac Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA (adjust for your CUDA version)
# CUDA 11.8:
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1:
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# Download models
python models/download_models.py

# Start server
python main.py --host 0.0.0.0 --port 8001
```

### Option 3: Docker (Recommended for Production)

```bash
# Build and run with GPU support
docker-compose up -d

# Or build manually
docker build -t f5tts-api .
docker run --gpus all -p 8001:8001 f5tts-api
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Server status |
| GET | `/status` | Detailed status with GPU info |
| GET | `/health` | Health check |
| POST | `/generate` | Generate speech |
| GET | `/docs` | Swagger API documentation |
| GET | `/redoc` | ReDoc API documentation |

### GET /status

Check server status and GPU availability.

**Response:**
```json
{
    "status": "running",
    "model_loaded": true,
    "gpu_available": true,
    "gpu_name": "NVIDIA GeForce RTX 3090",
    "version": "1.0.0"
}
```

### POST /generate

Generate speech from text using voice cloning.

**Request:**
```json
{
    "text": "Hello, this is a test of voice cloning.",
    "reference_audio": "<base64_encoded_audio>",
    "reference_text": "Optional transcript of reference audio",
    "speed": 1.0,
    "nfe_step": 32,
    "cfg_strength": 2.0,
    "sway_sampling_coef": -1.0,
    "language": "multilingual",
    "clean_audio": true,
    "noise_reduction_strength": 0.3,
    "remove_silence": false
}
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| text | string | required | Text to synthesize |
| reference_audio | string | required | Base64 encoded WAV/MP3 |
| reference_text | string | "" | Transcript of reference |
| speed | float | 1.0 | Speech speed (0.5-2.0) |
| nfe_step | int | 32 | NFE steps (8-128) |
| cfg_strength | float | 2.0 | CFG strength (0-5) |
| language | string | "multilingual" | Language code |
| clean_audio | bool | true | Apply noise reduction |
| remove_silence | bool | false | Remove silence |

**Response:**
```json
{
    "success": true,
    "audio_data": "<base64_encoded_wav>",
    "duration": 2.5,
    "sample_rate": 24000,
    "file_size": 120000
}
```

## ⚙️ Configuration

### Environment Variables

Create `.env` file from `.env.example`:

```bash
# Server Settings
HOST=0.0.0.0
PORT=8001

# GPU Settings
CUDA_VISIBLE_DEVICES=0

# Logging
LOG_LEVEL=INFO
```

## 🔗 Integration with Voice Cloning Platform

In your Django Voice Cloning platform, set these environment variables in `.env`:

```bash
TTS_API_URL=http://your-gpu-server-ip:8001/generate
TTS_API_KEY=
TTS_API_TIMEOUT=300
```

### Architecture Diagram

```
┌─────────────────────────────┐         ┌─────────────────────────────┐
│                             │         │                             │
│   Voice Cloning Platform    │  HTTP   │   F5-TTS API Server         │
│   (Django - Port 8000)      │ ──────► │   (FastAPI - Port 8001)     │
│                             │         │                             │
│   - User Management         │         │   - F5-TTS Model            │
│   - Credits System          │         │   - Voice Generation        │
│   - Payment Processing      │         │   - Audio Processing        │
│   - Admin Dashboard         │         │                             │
│                             │         │                             │
└─────────────────────────────┘         └─────────────────────────────┘
         │                                        │
         │                                        │
    Web Server                              GPU Server
    (CPU - No GPU needed)                   (NVIDIA GPU)
```

## 📊 Performance

| Metric | Value |
|--------|-------|
| First request | 30-60 seconds (model loading) |
| Short text (<100 chars) | 2-5 seconds |
| Long text (1000+ chars) | 10-30 seconds |
| Max text length | 50,000 characters |

**Tips for better performance:**
- Keep server running (avoid cold starts)
- Use SSD for model storage
- More VRAM = faster generation
- Reduce `nfe_step` for faster (lower quality) output

## 🔧 Troubleshooting

### CUDA out of memory
```bash
# Reduce NFE steps
"nfe_step": 16  # Instead of 32

# Or process shorter texts
# Split long text into chunks
```

### Model not loading
```bash
# Check CUDA installation
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Connection refused
```bash
# Check if server is running
curl http://localhost:8001/health

# Check firewall rules
# Allow port 8001 in firewall
```

### Slow generation on GPU
```bash
# Check if GPU is being used
# Look for "Using GPU: ..." in server logs

# If using CPU, reinstall PyTorch with CUDA:
pip uninstall torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📝 Example Usage (Python)

```python
import requests
import base64

# Read reference audio
with open("reference.wav", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode()

# Generate speech
response = requests.post(
    "http://localhost:8001/generate",
    json={
        "text": "Hello, this is a voice cloning test.",
        "reference_audio": audio_base64,
        "reference_text": "This is the reference audio transcript.",
        "speed": 1.0,
        "nfe_step": 32
    },
    timeout=300
)

result = response.json()
if result["success"]:
    # Decode and save output
    audio_bytes = base64.b64decode(result["audio_data"])
    with open("output.wav", "wb") as f:
        f.write(audio_bytes)
    print(f"Generated {result['duration']:.2f}s audio")
else:
    print(f"Error: {result['error']}")
```

## 📄 License

This project uses F5-TTS model. Please check the original license:
- F5-TTS: https://github.com/SWivid/F5-TTS

## 🔗 Links

- F5-TTS GitHub: https://github.com/SWivid/F5-TTS
- F5-TTS HuggingFace: https://huggingface.co/SWivid/F5-TTS
- FastAPI Docs: https://fastapi.tiangolo.com/
