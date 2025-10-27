# Landmark Identification API (.NET 8.0)

REST API for Vilnius landmark identification using TensorFlow Lite INT8 quantized model.

## Overview

- **Framework**: ASP.NET Core 8.0 Web API
- **Model**: TensorFlow Lite INT8 (1.12 MB)
- **Image Processing**: SixLabors.ImageSharp
- **Inference**: Python TFLite interpreter (subprocess bridge)
- **Landmarks**: 5 Vilnius landmarks (gediminas_tower, vilnius_cathedral, gate_of_dawn, st_anne, three_crosses)

## Architecture

The API uses a hybrid approach:
- **.NET Core** handles HTTP requests, image preprocessing, and API logic
- **Python TFLite** performs model inference via subprocess
- This avoids .NET TFLite library compatibility issues while maintaining good performance

### Production Alternatives
For production deployment, consider:
1. Convert model to ONNX format and use `Microsoft.ML.OnnxRuntime`
2. Use a dedicated Python microservice with FastAPI
3. Deploy TFLite model to mobile clients directly (Flutter)

## API Endpoints

### POST `/api/prediction/predict`
Predict landmark from uploaded image.

**Request:**
- Content-Type: `multipart/form-data`
- Field: `imageFile` (JPG/PNG, max 10MB)

**Response:**
```json
{
  "predictions": [
    {
      "label": "three_crosses",
      "confidence": 0.2897,
      "rank": 1
    },
    {
      "label": "gate_of_dawn",
      "confidence": 0.2391,
      "rank": 2
    },
    {
      "label": "st_anne",
      "confidence": 0.2376,
      "rank": 3
    }
  ],
  "inferenceTimeMs": 4037
}
```

### GET `/api/prediction/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-27T16:28:06.166766Z"
}
```

## Setup & Running

### Prerequisites
- .NET 8.0 SDK
- Python 3.11+ with TensorFlow (for inference)
- Virtual environment at: `/Users/rokas/Documents/KTU AI Masters/Team Project/.venv/`

### Build
```bash
cd backend/backend/LandmarkApi
dotnet build
```

### Run
```bash
dotnet run
```

API will start on: `http://localhost:5126`
Swagger UI: `http://localhost:5126/swagger`

## Testing

### cURL Example
```bash
curl -X POST http://localhost:5126/api/prediction/predict \
  -F "imageFile=@/path/to/image.jpg" \
  -H "Content-Type: multipart/form-data"
```

### Swagger UI
Navigate to `http://localhost:5126/swagger` for interactive API documentation.

## Configuration

Edit `appsettings.json` or use environment variables:

```json
{
  "ModelPath": "Models/landmark_mnv3_int8_drq.tflite",
  "LabelsPath": "Models/labels.txt",
  "PythonPath": "/path/to/.venv/bin/python"
}
```

## Project Structure

```
LandmarkApi/
├── Controllers/
│   └── PredictionController.cs    # API endpoints
├── Services/
│   └── LandmarkPredictionService.cs  # Prediction logic
├── Models/
│   ├── landmark_mnv3_int8_drq.tflite  # TFLite model
│   └── labels.txt                      # Class labels
├── Program.cs                     # App configuration
└── LandmarkApi.csproj            # Project file
```

## Dependencies

- **SixLabors.ImageSharp** 3.1.11 - Image processing
- **Microsoft.ML** 4.0.2 - ML framework
- **Microsoft.ML.TensorFlow** 4.0.2 - TensorFlow integration

## Performance

- **Model Size**: 1.12 MB (INT8 quantized)
- **Inference Time**: ~50-150ms (device-dependent)
- **Top-1 Accuracy**: ~77-78% (validation set)
- **Top-3 Accuracy**: ~96-98% (validation set)

## CORS

CORS is enabled for all origins in development mode. For production, update the CORS policy in `Program.cs`:

```csharp
options.AddPolicy("Production", policy =>
{
    policy.WithOrigins("https://yourdomain.com")
          .AllowAnyMethod()
          .AllowAnyHeader();
});
```

## Sprint 2 Integration

This API is ready for Flutter app integration:
1. Use HTTP client to POST images to `/api/prediction/predict`
2. Display Top-3 results with confidence scores
3. Handle errors (network, invalid image, etc.)
4. Add loading indicators during inference

Example Flutter HTTP request:
```dart
var request = http.MultipartRequest(
  'POST',
  Uri.parse('http://localhost:5126/api/prediction/predict'),
);
request.files.add(
  await http.MultipartFile.fromPath('imageFile', imagePath)
);
var response = await request.send();
```

## License

See main project README for licensing information.
