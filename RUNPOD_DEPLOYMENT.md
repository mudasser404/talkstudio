# RunPod Serverless Deployment Guide - F5-TTS Voice Cloning API

## Quick Start

### 1. Build Docker Image

```bash
# Build the image
docker build -t f5tts-runpod:latest .

# Tag for Docker Hub (replace 'yourusername')
docker tag f5tts-runpod:latest yourusername/f5tts-runpod:latest

# Push to Docker Hub
docker push yourusername/f5tts-runpod:latest
```

### 2. Create RunPod Serverless Endpoint

1. Go to [RunPod Console](https://www.runpod.io/console/serverless)
2. Click "New Endpoint"
3. Configure:
   - **Container Image**: `yourusername/f5tts-runpod:latest`
   - **GPU Type**: RTX 3090, RTX 4090, or A100 (recommended)
   - **Min Workers**: 0 (scales to zero when idle)
   - **Max Workers**: 3-5 (based on your needs)
   - **Idle Timeout**: 5 seconds
   - **Execution Timeout**: 300 seconds
   - **Volume**: Enable Network Volume for model caching (optional but recommended)

### 3. API Usage

#### Get Endpoint URL
After creating the endpoint, you'll get a URL like:
```
https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync
```

#### Authentication
Use your RunPod API Key in the header:
```
Authorization: Bearer YOUR_RUNPOD_API_KEY
```

---

## API Endpoints

### Generate Speech (Synchronous)

**POST** `/runsync`

```json
{
    "input": {
        "action": "generate",
        "text": "Hello, this is a test of voice cloning.",
        "reference_audio": "BASE64_ENCODED_AUDIO",
        "reference_text": "Optional transcript of reference audio",
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
    "id": "job-id",
    "status": "COMPLETED",
    "output": {
        "success": true,
        "audio_data": "BASE64_ENCODED_OUTPUT_AUDIO",
        "duration": 3.5,
        "sample_rate": 24000,
        "file_size": 168000,
        "seed": 12345678
    }
}
```

### Batch Generate (Multiple texts)

```json
{
    "input": {
        "action": "batch_generate",
        "texts": [
            "First sentence to generate.",
            "Second sentence to generate.",
            "Third sentence to generate."
        ],
        "reference_audio": "BASE64_ENCODED_AUDIO",
        "reference_text": "Reference audio transcript",
        "speed": 1.0,
        "nfe_step": 32
    }
}
```

### Check Status

```json
{
    "input": {
        "action": "status"
    }
}
```

---

## Python Client Example

```python
import requests
import base64
import json

RUNPOD_API_KEY = "your_api_key"
ENDPOINT_ID = "your_endpoint_id"

def generate_speech(text, reference_audio_path, reference_text=""):
    # Read and encode reference audio
    with open(reference_audio_path, 'rb') as f:
        reference_audio_b64 = base64.b64encode(f.read()).decode('utf-8')

    # Make request
    response = requests.post(
        f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync",
        headers={
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "input": {
                "action": "generate",
                "text": text,
                "reference_audio": reference_audio_b64,
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
            audio_bytes = base64.b64decode(output["audio_data"])
            with open("output.wav", 'wb') as f:
                f.write(audio_bytes)
            print(f"Audio saved! Duration: {output['duration']:.2f}s")
            return True

    print(f"Error: {result}")
    return False

# Usage
generate_speech(
    text="Hello, this is a voice cloning test.",
    reference_audio_path="reference.wav",
    reference_text="This is the reference audio transcript."
)
```

---

## JavaScript/Node.js Client Example

```javascript
const axios = require('axios');
const fs = require('fs');

const RUNPOD_API_KEY = 'your_api_key';
const ENDPOINT_ID = 'your_endpoint_id';

async function generateSpeech(text, referenceAudioPath, referenceText = '') {
    // Read and encode reference audio
    const audioBuffer = fs.readFileSync(referenceAudioPath);
    const referenceAudioB64 = audioBuffer.toString('base64');

    try {
        const response = await axios.post(
            `https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync`,
            {
                input: {
                    action: 'generate',
                    text: text,
                    reference_audio: referenceAudioB64,
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
            const audioBuffer = Buffer.from(result.output.audio_data, 'base64');
            fs.writeFileSync('output.wav', audioBuffer);
            console.log(`Audio saved! Duration: ${result.output.duration.toFixed(2)}s`);
            return true;
        }

        console.error('Error:', result);
        return false;

    } catch (error) {
        console.error('Request failed:', error.message);
        return false;
    }
}

// Usage
generateSpeech(
    'Hello, this is a voice cloning test.',
    'reference.wav',
    'This is the reference audio transcript.'
);
```

---

## Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required | Text to synthesize |
| `reference_audio` | string | required | Base64 encoded reference audio (WAV/MP3) |
| `reference_text` | string | "" | Transcript of reference audio (improves quality) |
| `speed` | float | 1.0 | Speech speed (0.5-2.0) |
| `nfe_step` | int | 32 | Number of inference steps (8-128, higher = better quality) |
| `cfg_strength` | float | 2.0 | Classifier-free guidance strength (0-5) |
| `sway_sampling_coef` | float | -1.0 | Sway sampling coefficient |
| `seed` | int | null | Random seed for reproducibility |

---

## Cost Optimization Tips

1. **Use Network Volume**: Store model cache on network volume to reduce cold start time
2. **Batch Requests**: Use `batch_generate` for multiple texts
3. **Optimize `nfe_step`**: Lower values (16-24) are faster but slightly lower quality
4. **Right-size GPU**: RTX 3090 is usually sufficient, A100 for high throughput
5. **Set appropriate timeouts**: Most requests complete in 5-30 seconds

---

## Troubleshooting

### Cold Start Taking Too Long
- Enable model pre-loading in Dockerfile (uncomment the pre-download line)
- Use network volume for model caching

### Out of Memory
- Reduce `nfe_step` value
- Use shorter reference audio (<12 seconds)
- Split long text into smaller chunks

### Audio Quality Issues
- Increase `nfe_step` to 48-64
- Provide accurate `reference_text`
- Use high-quality reference audio (clean, no background noise)