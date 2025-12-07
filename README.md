# F5-TTS Voice Cloning API Server

Standalone FastAPI server for F5-TTS voice cloning model. Deploy on a GPU server locally or on **RunPod Serverless**.

## 📁 Project Structure

```
f5tts_api_server/
├── main.py                    # FastAPI server (local deployment)
├── handler.py                 # RunPod Serverless handler
├── config.py                  # Configuration settings
├── requirements.txt           # Python dependencies (local)
├── requirements_runpod.txt    # RunPod dependencies
├── Dockerfile                 # RunPod Docker image
├── RUNPOD_DEPLOYMENT.md       # Detailed RunPod guide
├── F5_TTS_API_Collection.json # Postman collection
├── api_examples.http          # HTTP examples
├── templates/
│   └── index.html             # Web UI
├── routers/
│   ├── tts.py                 # TTS endpoints
│   ├── training.py            # Training endpoints
│   └── dataset.py             # Dataset endpoints
├── services/
│   ├── tts_service.py         # TTS logic
│   ├── training_service.py    # Training logic
│   └── dataset_service.py     # Dataset logic
├── models/
│   └── schemas.py             # Pydantic models
└── F5-TTS/                    # F5-TTS model source
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

---

# 🚀 Deployment Options

## Option 1: RunPod Serverless (Recommended for Production)

RunPod Serverless provides pay-per-use GPU computing - you only pay when generating audio.

### Step 1: Install Docker Desktop

Download and install Docker Desktop from: https://www.docker.com/products/docker-desktop/

### Step 2: Build Docker Image

```bash
cd f5tts_api_server

# Build the image
docker build -t f5tts-runpod:latest .
```

### Step 3: Push to Docker Hub

```bash
# Login to Docker Hub (create account at hub.docker.com)
docker login

# Tag the image (replace YOUR_USERNAME with your Docker Hub username)
docker tag f5tts-runpod:latest YOUR_USERNAME/f5tts-runpod:latest

# Push to Docker Hub
docker push YOUR_USERNAME/f5tts-runpod:latest
```

### Step 4: Create RunPod Account & Endpoint

1. Go to [RunPod](https://www.runpod.io/) and create an account
2. Add credits to your account (minimum $10 recommended)
3. Go to [Serverless Console](https://www.runpod.io/console/serverless)
4. Click **"New Endpoint"**
5. Configure settings:

| Setting | Value |
|---------|-------|
| **Container Image** | `YOUR_USERNAME/f5tts-runpod:latest` |
| **GPU Type** | RTX 3090 or RTX 4090 |
| **Min Workers** | 0 (scales to zero when idle) |
| **Max Workers** | 3-5 |
| **Idle Timeout** | 5 seconds |
| **Execution Timeout** | 300 seconds |
| **Volume** | Enable Network Volume (optional) |

6. Click **"Create Endpoint"**
7. Copy your **Endpoint ID** (e.g., `abc123xyz`)

### Step 5: Get API Key

1. Go to [RunPod Settings](https://www.runpod.io/console/user/settings)
2. Click **"API Keys"**
3. Create a new API key and copy it

### Step 6: Test Your Endpoint

**Check Status:**
```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"action": "status"}}'
```

**Generate Speech:**
```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "text": "Hello, this is a voice cloning test.",
      "reference_audio_url": "https://example.com/reference.wav",
      "reference_text": "Reference audio transcript",
      "speed": 1.0,
      "nfe_step": 32
    }
  }'
```

### RunPod API Reference

**Endpoint URL:**
```
https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync
```

**Headers:**
```
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json
```

**Request Body:**
```json
{
    "input": {
        "text": "Text to synthesize",
        "reference_audio_url": "https://example.com/audio.wav",
        "reference_text": "Transcript of reference audio",
        "speed": 1.0,
        "nfe_step": 32,
        "cfg_strength": 2.0,
        "sway_sampling_coef": -1.0,
        "seed": null
    }
}
```

**Response:**
```json
{
    "id": "runpod-job-id",
    "status": "COMPLETED",
    "output": {
        "success": true,
        "job_id": "unique-job-uuid",
        "status": "completed",
        "audio_base64": "BASE64_ENCODED_WAV_AUDIO",
        "duration": 3.5,
        "sample_rate": 24000,
        "file_size": 168000,
        "seed": 12345678,
        "started_at": "2024-01-01T00:00:00",
        "completed_at": "2024-01-01T00:00:05",
        "parameters": {
            "text": "...",
            "reference_audio_url": "...",
            "reference_text": "...",
            "speed": 1.0,
            "nfe_step": 32,
            "cfg_strength": 2.0,
            "sway_sampling_coef": -1.0
        }
    }
}
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required | Text to synthesize |
| `reference_audio_url` | string | required | URL of reference audio (WAV/MP3) |
| `reference_text` | string | "" | Transcript of reference audio |
| `speed` | float | 1.0 | Speech speed (0.5-2.0) |
| `nfe_step` | int | 32 | Inference steps (8-128, higher = better) |
| `cfg_strength` | float | 2.0 | Classifier-free guidance (0-5) |
| `sway_sampling_coef` | float | -1.0 | Sway sampling coefficient |
| `seed` | int | null | Random seed for reproducibility |

### Python Client Example (RunPod)

```python
import requests
import base64
import json

RUNPOD_API_KEY = "your_api_key"
ENDPOINT_ID = "your_endpoint_id"

def generate_speech(text, reference_audio_url, reference_text=""):
    """Generate speech using RunPod F5-TTS endpoint"""

    response = requests.post(
        f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync",
        headers={
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "input": {
                "text": text,
                "reference_audio_url": reference_audio_url,
                "reference_text": reference_text,
                "speed": 1.0,
                "nfe_step": 32,
                "cfg_strength": 2.0
            }
        },
        timeout=300
    )

    result = response.json()

    if result.get("status") == "COMPLETED":
        output = result["output"]
        if output.get("success"):
            # Decode and save audio
            audio_bytes = base64.b64decode(output["audio_base64"])
            with open("output.wav", 'wb') as f:
                f.write(audio_bytes)

            print(f"✅ Audio saved!")
            print(f"   Job ID: {output['job_id']}")
            print(f"   Duration: {output['duration']:.2f}s")
            print(f"   Sample Rate: {output['sample_rate']}")
            return output
        else:
            print(f"❌ Error: {output.get('error')}")
    else:
        print(f"❌ Job failed: {result}")

    return None

# Usage
generate_speech(
    text="Hello, this is a voice cloning test.",
    reference_audio_url="https://your-server.com/reference.wav",
    reference_text="This is the reference audio transcript."
)
```

### JavaScript/Node.js Client Example (RunPod)

```javascript
const axios = require('axios');
const fs = require('fs');

const RUNPOD_API_KEY = 'your_api_key';
const ENDPOINT_ID = 'your_endpoint_id';

async function generateSpeech(text, referenceAudioUrl, referenceText = '') {
    try {
        const response = await axios.post(
            `https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync`,
            {
                input: {
                    text: text,
                    reference_audio_url: referenceAudioUrl,
                    reference_text: referenceText,
                    speed: 1.0,
                    nfe_step: 32,
                    cfg_strength: 2.0
                }
            },
            {
                headers: {
                    'Authorization': `Bearer ${RUNPOD_API_KEY}`,
                    'Content-Type': 'application/json'
                },
                timeout: 300000
            }
        );

        const result = response.data;

        if (result.status === 'COMPLETED' && result.output.success) {
            const audioBuffer = Buffer.from(result.output.audio_base64, 'base64');
            fs.writeFileSync('output.wav', audioBuffer);

            console.log('✅ Audio saved!');
            console.log(`   Job ID: ${result.output.job_id}`);
            console.log(`   Duration: ${result.output.duration.toFixed(2)}s`);
            return result.output;
        }

        console.error('❌ Error:', result);
        return null;

    } catch (error) {
        console.error('❌ Request failed:', error.message);
        return null;
    }
}

// Usage
generateSpeech(
    'Hello, this is a voice cloning test.',
    'https://your-server.com/reference.wav',
    'This is the reference audio transcript.'
);
```

### RunPod Cost Optimization Tips

1. **Set Min Workers to 0**: Only pay when actually processing requests
2. **Use Network Volume**: Reduces cold start time by caching model
3. **Batch Requests**: Generate multiple audio files per API call
4. **Optimize nfe_step**: Use 16-24 for faster (slightly lower quality) results
5. **Right-size GPU**: RTX 3090 is usually sufficient

---

## Option 2: Local Server

### Windows Setup

```bash
# Step 1: Run setup
setup.bat

# Step 2: Download models
download_model.bat

# Step 3: Start server
start_server.bat
```

### Linux/Mac Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA
# CUDA 11.8:
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1:
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# Install F5-TTS
pip install -e ./F5-TTS

# Start server
python main.py --host 0.0.0.0 --port 8001
```

### Local API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Server status |
| GET | `/status` | Detailed status with GPU info |
| GET | `/health` | Health check |
| GET | `/ui` | Web UI |
| GET | `/docs` | Swagger API docs |
| POST | `/api/tts/generate` | Generate speech |
| POST | `/api/tts/model/load` | Load model |
| POST | `/api/training/*` | Training endpoints |
| POST | `/api/datasets/*` | Dataset endpoints |

---

## 🔗 Integration with Django Voice Cloning Platform

### For RunPod Deployment

In your Django `.env` file:
```bash
TTS_API_TYPE=runpod
RUNPOD_ENDPOINT_ID=your_endpoint_id
RUNPOD_API_KEY=your_api_key
```

### For Local Deployment

In your Django `.env` file:
```bash
TTS_API_TYPE=local
TTS_API_URL=http://your-gpu-server-ip:8001
```

### Architecture

```
┌─────────────────────────────┐         ┌─────────────────────────────┐
│                             │         │                             │
│   Voice Cloning Platform    │  HTTP   │   F5-TTS API                │
│   (Django - Port 8000)      │ ──────► │                             │
│                             │         │   Option A: RunPod          │
│   - User Management         │         │   (Serverless GPU)          │
│   - Credits System          │         │                             │
│   - Payment Processing      │         │   Option B: Local Server    │
│   - Admin Dashboard         │         │   (Your GPU Server)         │
│                             │         │                             │
└─────────────────────────────┘         └─────────────────────────────┘
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Cold start (first request) | 30-60 seconds |
| Short text (<100 chars) | 2-5 seconds |
| Long text (1000+ chars) | 10-30 seconds |
| Max text length | 50,000 characters |

---

## 🔧 Troubleshooting

### RunPod Issues

**Cold Start Taking Too Long:**
- Enable Network Volume for model caching
- Pre-download model in Dockerfile (uncomment the pre-download line)

**Out of Memory on RunPod:**
- Reduce `nfe_step` to 16-24
- Use shorter reference audio (<12 seconds)
- Split long text into smaller chunks

**Job Timeout:**
- Increase Execution Timeout in endpoint settings
- Split very long text into chunks

### Local Server Issues

**CUDA out of memory:**
```bash
# Reduce NFE steps
"nfe_step": 16  # Instead of 32
```

**Model not loading:**
```bash
# Check CUDA installation
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

**Connection refused:**
```bash
# Check if server is running
curl http://localhost:8001/health

# Check firewall - allow port 8001
```

---

## 📝 Files Reference

| File | Purpose |
|------|---------|
| `handler.py` | RunPod serverless handler |
| `Dockerfile` | RunPod Docker image |
| `requirements_runpod.txt` | RunPod dependencies |
| `RUNPOD_DEPLOYMENT.md` | Detailed RunPod guide |
| `main.py` | Local FastAPI server |
| `requirements.txt` | Local dependencies |
| `F5_TTS_API_Collection.json` | Postman collection |
| `api_examples.http` | HTTP examples |

---

## 📄 License

This project uses F5-TTS model. Please check the original license:
- F5-TTS: https://github.com/SWivid/F5-TTS

## 🔗 Links

- F5-TTS GitHub: https://github.com/SWivid/F5-TTS
- F5-TTS HuggingFace: https://huggingface.co/SWivid/F5-TTS
- RunPod: https://www.runpod.io/
- FastAPI Docs: https://fastapi.tiangolo.com/