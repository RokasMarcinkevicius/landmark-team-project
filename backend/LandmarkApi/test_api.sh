#!/bin/bash
# Quick API test script

API_URL="http://localhost:5126"

echo "========================================="
echo "Landmark Identification API - Test Script"
echo "========================================="
echo ""

# Check if server is running
echo "1. Testing health endpoint..."
health_response=$(curl -s "$API_URL/api/prediction/health")
if [ $? -eq 0 ]; then
    echo "✓ Health check: $health_response"
else
    echo "✗ Server not running. Start with: dotnet run"
    exit 1
fi

echo ""
echo "2. Creating test image..."
python3 - << 'EOF'
from PIL import Image
img = Image.new('RGB', (224, 224), color=(100, 150, 200))
img.save('/tmp/test_landmark.jpg')
print("✓ Test image created")
EOF

echo ""
echo "3. Testing prediction endpoint..."
curl -X POST "$API_URL/api/prediction/predict" \
  -F "imageFile=@/tmp/test_landmark.jpg" \
  -H "Content-Type: multipart/form-data" \
  -s | python3 -m json.tool

echo ""
echo "========================================="
echo "Test complete!"
echo "Swagger UI: $API_URL/swagger"
echo "========================================="
