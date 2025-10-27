# iOS App Requirements for Landmark Identification

## Overview
Build a native iOS app for Vilnius landmark identification using the .NET backend API.

---

## 1. Backend API (Ready ✅)

### Base URL
```
http://localhost:5126 (development)
https://your-server.com (production)
```

### Endpoints

#### POST /api/prediction/predict
Upload image and get Top-3 landmark predictions.

**Request:**
```http
POST /api/prediction/predict
Content-Type: multipart/form-data

imageFile: <binary data>
```

**Response:**
```json
{
  "predictions": [
    {
      "label": "gediminas_tower",
      "confidence": 0.8234,
      "rank": 1
    },
    {
      "label": "vilnius_cathedral",
      "confidence": 0.1234,
      "rank": 2
    },
    {
      "label": "gate_of_dawn",
      "confidence": 0.0432,
      "rank": 3
    }
  ],
  "inferenceTimeMs": 156
}
```

#### GET /api/prediction/history?limit=20
Get recent prediction history.

**Response:**
```json
[
  {
    "id": 1,
    "originalFilename": "IMG_1234.jpg",
    "timestamp": "2025-10-27T16:30:00Z",
    "topPrediction": "gediminas_tower",
    "topConfidence": 0.8234,
    "inferenceTimeMs": 156,
    "imageSizeMb": 2.3
  }
]
```

---

## 2. iOS App Requirements

### Minimum Requirements
- **iOS Version:** iOS 15.0+
- **Xcode:** 14.0+ or latest
- **Language:** Swift 5.7+
- **Architecture:** SwiftUI or UIKit

### Core Features

#### 2.1 Camera Integration
- [ ] Access device camera (front/back)
- [ ] Capture photo capability
- [ ] Photo library access (import existing photos)
- [ ] Real-time camera preview
- [ ] Flash control

**Permissions needed:**
```xml
<!-- Info.plist -->
<key>NSCameraUsageDescription</key>
<string>We need camera access to identify Vilnius landmarks</string>
<key>NSPhotoLibraryUsageDescription</key>
<string>We need photo library access to select landmark images</string>
```

#### 2.2 Image Upload & API Integration
- [ ] HTTP client for API calls (URLSession or Alamofire)
- [ ] Multipart form-data upload
- [ ] Image compression before upload (max 10MB)
- [ ] Loading indicators during upload/inference
- [ ] Error handling (network, timeout, server errors)
- [ ] Retry logic

**Swift Example:**
```swift
func uploadImage(_ image: UIImage) async throws -> PredictionResult {
    guard let imageData = image.jpegData(compressionQuality: 0.8) else {
        throw ImageError.compressionFailed
    }

    var request = URLRequest(url: URL(string: "\(baseURL)/api/prediction/predict")!)
    request.httpMethod = "POST"

    let boundary = UUID().uuidString
    request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

    var body = Data()
    body.append("--\(boundary)\r\n".data(using: .utf8)!)
    body.append("Content-Disposition: form-data; name=\"imageFile\"; filename=\"photo.jpg\"\r\n".data(using: .utf8)!)
    body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
    body.append(imageData)
    body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)

    request.httpBody = body

    let (data, response) = try await URLSession.shared.data(for: request)

    guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
        throw NetworkError.invalidResponse
    }

    return try JSONDecoder().decode(PredictionResult.swift, from: data)
}
```

#### 2.3 Results Display
- [ ] Show Top-3 predictions with confidence scores
- [ ] Display landmark names (user-friendly format)
- [ ] Confidence visualization (progress bars, percentages)
- [ ] Inference time display
- [ ] "Unsure" threshold handling (< 55% confidence)
- [ ] Landmark information cards
- [ ] Share result capability

**Landmark Labels (Map to Display Names):**
```swift
let landmarkNames: [String: String] = [
    "gediminas_tower": "Gediminas Tower",
    "vilnius_cathedral": "Vilnius Cathedral",
    "gate_of_dawn": "Gate of Dawn",
    "st_anne": "St. Anne's Church",
    "three_crosses": "Three Crosses Monument"
]
```

#### 2.4 History View
- [ ] List recent predictions
- [ ] Thumbnail preview
- [ ] Tap to view details
- [ ] Pull-to-refresh
- [ ] Pagination (load more)

#### 2.5 Settings
- [ ] API base URL configuration
- [ ] Camera quality settings
- [ ] Upload quality settings
- [ ] Clear history
- [ ] About/Credits

---

## 3. Technical Stack Recommendations

### Networking
- **URLSession** (built-in) - Recommended for simplicity
- **Alamofire** (optional) - For advanced features

### UI Framework
- **SwiftUI** - Modern, declarative UI
- **UIKit** - Traditional, more control

### Image Handling
- **UIKit.UIImage** - Built-in
- **Kingfisher** (optional) - For image caching

### Data Persistence
- **UserDefaults** - Simple settings
- **CoreData** or **SwiftData** (optional) - Local history cache

### Architecture
- **MVVM** (Model-View-ViewModel) - Recommended
- **MVC** - Traditional approach

---

## 4. Project Structure

```
LandmarkApp/
├── Models/
│   ├── PredictionResult.swift
│   ├── LandmarkPrediction.swift
│   └── HistoryRecord.swift
├── Views/
│   ├── CameraView.swift
│   ├── ResultsView.swift
│   ├── HistoryView.swift
│   └── SettingsView.swift
├── ViewModels/
│   ├── CameraViewModel.swift
│   ├── PredictionViewModel.swift
│   └── HistoryViewModel.swift
├── Services/
│   ├── APIClient.swift
│   ├── CameraService.swift
│   └── ImageProcessor.swift
├── Utilities/
│   ├── Extensions.swift
│   └── Constants.swift
└── Resources/
    ├── Assets.xcassets
    └── Info.plist
```

---

## 5. Data Models (Swift)

```swift
struct PredictionResult: Codable {
    let predictions: [LandmarkPrediction]
    let inferenceTimeMs: Int
}

struct LandmarkPrediction: Codable {
    let label: String
    let confidence: Float
    let rank: Int

    var displayName: String {
        landmarkNames[label] ?? label.capitalized
    }

    var confidencePercentage: String {
        String(format: "%.1f%%", confidence * 100)
    }
}

struct HistoryRecord: Codable, Identifiable {
    let id: Int
    let originalFilename: String
    let timestamp: Date
    let topPrediction: String
    let topConfidence: Float
    let inferenceTimeMs: Int
    let imageSizeMb: Double
}
```

---

## 6. Key Features Implementation

### 6.1 Camera Capture (SwiftUI Example)

```swift
import SwiftUI
import AVFoundation

struct CameraView: View {
    @StateObject private var camera = CameraService()
    @State private var showResults = false

    var body: some View {
        ZStack {
            CameraPreview(camera: camera)
                .edgesIgnoringSafeArea(.all)

            VStack {
                Spacer()
                Button(action: camera.capturePhoto) {
                    Circle()
                        .fill(Color.white)
                        .frame(width: 70, height: 70)
                }
                .padding(.bottom, 30)
            }
        }
        .onAppear {
            camera.checkPermissions()
        }
    }
}
```

### 6.2 "Unsure" Threshold

```swift
extension PredictionResult {
    var isConfident: Bool {
        guard let topPrediction = predictions.first else { return false }
        return topPrediction.confidence >= 0.55
    }

    var confidenceMessage: String {
        if isConfident {
            return "High confidence"
        } else {
            return "Low confidence - Try taking another photo"
        }
    }
}
```

---

## 7. Database Setup (Backend)

The backend requires PostgreSQL. Set it up with:

### Option 1: Local PostgreSQL
```bash
# Install PostgreSQL (macOS)
brew install postgresql
brew services start postgresql

# Create database
createdb landmark_db

# Update appsettings.json connection string
```

### Option 2: Docker PostgreSQL
```bash
docker run --name landmark-postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=landmark_db \
  -p 5432:5432 \
  -d postgres:15

# Backend will auto-migrate tables on startup
```

### Option 3: Cloud PostgreSQL
- **Heroku Postgres** (free tier available)
- **AWS RDS** (PostgreSQL)
- **DigitalOcean Managed Databases**
- **Supabase** (free tier with API)

---

## 8. Deployment Checklist

### Backend
- [ ] Set up PostgreSQL database
- [ ] Update connection string in `appsettings.json`
- [ ] Deploy API to cloud (Azure, AWS, Heroku, DigitalOcean)
- [ ] Update `uploads/` directory permissions
- [ ] Configure HTTPS
- [ ] Update CORS for production domain

### iOS App
- [ ] Update API base URL to production
- [ ] Test all endpoints
- [ ] Add error handling for all network calls
- [ ] Test camera permissions
- [ ] Test on real devices (iPhone)
- [ ] Optimize images before upload
- [ ] Add app icons
- [ ] Configure App Store metadata
- [ ] Submit to App Store Connect

---

## 9. Testing Recommendations

### Unit Tests
- [ ] API client tests
- [ ] Image processing tests
- [ ] Prediction result parsing tests

### UI Tests
- [ ] Camera capture flow
- [ ] Upload and display results
- [ ] History list navigation
- [ ] Settings persistence

### Manual Testing
- [ ] Test with all 5 landmark types
- [ ] Test with poor lighting
- [ ] Test with low confidence scenarios
- [ ] Test offline error handling
- [ ] Test slow network conditions

---

## 10. Nice-to-Have Features (Optional)

- [ ] On-device TFLite model (offline mode)
- [ ] Map integration (show landmark location)
- [ ] AR feature (point camera at landmark for info)
- [ ] Multiple language support
- [ ] Dark mode support
- [ ] Accessibility features (VoiceOver)
- [ ] Analytics integration
- [ ] Push notifications
- [ ] Social sharing
- [ ] Favorites/Bookmarks

---

## 11. Resources

### Apple Documentation
- [Camera and Photos Framework](https://developer.apple.com/documentation/avfoundation/cameras_and_media_capture)
- [URLSession](https://developer.apple.com/documentation/foundation/urlsession)
- [SwiftUI](https://developer.apple.com/xcode/swiftui/)

### Tutorials
- [iOS Camera App Tutorial](https://www.hackingwithswift.com/books/ios-swiftui/how-to-let-users-select-pictures-using-photospicker)
- [Networking in Swift](https://www.hackingwithswift.com/books/ios-swiftui/sending-and-receiving-codable-data-with-urlsession-and-swiftui)

### Libraries
- [Alamofire](https://github.com/Alamofire/Alamofire) - HTTP networking
- [Kingfisher](https://github.com/onevcat/Kingfisher) - Image downloading/caching
- [SwiftUI Camera](https://github.com/rorodriguez116/SwiftUI-Camera) - Camera helper

---

## 12. Estimated Timeline

- **Week 1:** Project setup, API integration, basic UI
- **Week 2:** Camera implementation, image upload
- **Week 3:** Results display, history view
- **Week 4:** Polish, testing, bug fixes
- **Week 5:** Deployment, App Store submission

**Total:** 4-5 weeks for MVP

---

## 13. Contact & Support

- **Backend API:** http://localhost:5126/swagger (Swagger documentation)
- **Database:** PostgreSQL on localhost:5432
- **API Health Check:** http://localhost:5126/api/prediction/health

---

## Summary

**You have everything you need to build the iOS app:**
1. ✅ Working .NET backend API with image upload
2. ✅ PostgreSQL database for saving predictions
3. ✅ TFLite model (1.12 MB, 77-78% accuracy)
4. ✅ Complete API documentation
5. ✅ Swift code examples
6. ✅ Project structure recommendations

**Next steps:**
1. Set up PostgreSQL database
2. Test backend API (see test_api.sh)
3. Create new Xcode project
4. Implement camera capture
5. Integrate with API
6. Display results

Good luck building your iOS app! 🚀
