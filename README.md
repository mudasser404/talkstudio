# F5-TTS Voice Cloning API Server

Standalone FastAPI server for F5-TTS voice cloning model. Deploy this on a separate GPU server.

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (recommended: RTX 3090, A100, etc.)
- 8GB+ VRAM
- 16GB+ RAM

## Quick Start

### Option 1: Local Setup (Windows)

```bash
# Run setup script
setup.bat

# Start server
start_server.bat
```

### Option 2: Local Setup (Linux/Mac)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA (adjust for your CUDA version)
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Start server
python main.py --host 0.0.0.0 --port 8001
```

### Option 3: Docker

```bash
# Build and run with GPU support
docker-compose up -d

# Or build manually
docker build -t f5tts-api .
docker run --gpus all -p 8001:8001 f5tts-api
```

## API Endpoints

### GET /status
Check server status and GPU availability.

```json
{
    "status": "running",
    "model_loaded": true,
    "gpu_available": true,
    "gpu_name": "NVIDIA GeForce RTX 3090"
}
```

### POST /generate
Generate speech from text.

**Request:**
```json
{
    "text": "Hello, this is a test.",
    "reference_audio": "<base64_encoded_audio>",
    "reference_text": "Optional transcript",
    "speed": 1.0,
    "nfe_step": 32,
    "cfg_strength": 2.0,
    "language": "multilingual"
}
```

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

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device ID |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8001` | Server port |

## Integration with Voice Cloning Platform

In your Django Voice Cloning platform, set these environment variables:

```bash
TTS_API_URL=http://your-gpu-server:8001/generate
TTS_API_KEY=  # Optional, if you add authentication
TTS_API_TIMEOUT=300
```

## API Documentation

Interactive API docs available at:
- Swagger UI: `http://localhost:8001/docs`
- ReDoc: `http://localhost:8001/redoc`

## Performance

- First request may take 30-60 seconds (model loading)
- Subsequent requests: 2-10 seconds depending on text length
- Recommended: Keep server running for best performance

## Troubleshooting

### CUDA out of memory
- Reduce `nfe_step` (default: 32, min: 8)
- Process shorter texts
- Use a GPU with more VRAM

### Model not loading
- Check CUDA installation: `nvidia-smi`
- Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

### Slow generation
- Ensure GPU is being used (check logs)
- CPU mode is 10-50x slower than GPU
