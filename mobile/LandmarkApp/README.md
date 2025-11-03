# Landmark Identifier - React Native App

Mobile app for identifying Vilnius landmarks using your .NET backend API.

## Features

- 📷 **Camera Integration** - Take photos directly in the app
- 🖼️ **Gallery Picker** - Choose existing photos
- 🔍 **AI Predictions** - Get Top-3 landmark predictions with confidence scores
- 📊 **Visual Results** - See predictions with progress bars
- ⚠️ **Confidence Warnings** - Alert when predictions are uncertain (< 55%)
- 📜 **History** - View past predictions from the database
- 🔄 **Pull to Refresh** - Update history with latest data

## Screenshots

[Camera Screen] → [Take/Upload Photo] → [Results with Top-3] → [History List]

## Prerequisites

- Node.js 18+ 
- Expo Go app installed on your phone ([iOS](https://apps.apple.com/app/expo-go/id982107779) | [Android](https://play.google.com/store/apps/details?id=host.exp.exponent))
- Backend API running (see `backend/LandmarkApi/`)
- PostgreSQL database running

## Quick Start

### 1. Update API URL

Edit `src/constants/config.ts` and change the API URL to your computer's local IP:

```typescript
// Find your IP: ifconfig (Mac/Linux) or ipconfig (Windows)
export const API_BASE_URL = 'http://192.168.50.103:5126'; // Change this!
```

**Important:** `localhost` won't work on a physical device. Use your computer's IP address on the same WiFi network.

### 2. Start the Backend

```bash
# Terminal 1: Start PostgreSQL
docker run --name landmark-postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=landmark_db \
  -p 5432:5432 \
  -d postgres:15

# Terminal 2: Start API
cd ../backend/LandmarkApi
dotnet run
```

Backend will run on `http://localhost:5126`

### 3. Start the React Native App

```bash
nvm install 20
nvm use 20

cd mobile/LandmarkApp
npm start
```

### 4. Open on Your Phone

1. Open **Expo Go** app on your phone
2. Scan the QR code from the terminal
3. App will load on your phone
4. Grant camera and gallery permissions when prompted

## Project Structure

```
mobile/LandmarkApp/
├── App.tsx                           # Main navigation setup
├── src/
│   ├── screens/
│   │   ├── CameraScreen.tsx          # Camera/upload screen
│   │   ├── ResultsScreen.tsx         # Prediction results
│   │   └── HistoryScreen.tsx         # Prediction history
│   ├── services/
│   │   └── api.ts                    # API client (axios)
│   ├── types/
│   │   └── index.ts                  # TypeScript types
│   └── constants/
│       └── config.ts                 # API configuration
├── package.json
└── README.md
```

## Testing

### Test on Emulator/Simulator

**iOS Simulator (Mac only):**
```bash
npm run ios
```

**Android Emulator:**
```bash
npm run android
```

### Test on Physical Device

1. Install Expo Go on your phone
2. Make sure phone and computer are on same WiFi
3. Run `npm start`
4. Scan QR code with Expo Go

## API Configuration

The app calls these endpoints:

- `POST /api/prediction/predict` - Upload image for prediction
- `GET /api/prediction/history` - Get prediction history
- `GET /api/prediction/health` - Health check

### Testing Backend from Phone

To test if your phone can reach the backend:

1. Find your computer's IP: `ifconfig` (look for `en0` on Mac)
2. Open browser on phone
3. Navigate to: `http://YOUR_IP:5126/api/prediction/health`
4. Should see: `{"status":"healthy",...}`

If this doesn't work:
- Check firewall settings
- Make sure both devices are on same WiFi
- Try using `ngrok` to expose your API publicly

### Using ngrok (Alternative)

If same WiFi doesn't work, use ngrok:

```bash
# Install ngrok: https://ngrok.com/download
ngrok http 5126
```

Copy the `https://` URL and update `config.ts`:
```typescript
export const API_BASE_URL = 'https://abc123.ngrok.io';
```

## Troubleshooting

### "Cannot connect to server"

**Problem:** App can't reach backend API

**Solutions:**
1. Check backend is running: `curl http://localhost:5126/api/prediction/health`
2. Use correct IP address (not localhost)
3. Check firewall isn't blocking port 5126
4. Use ngrok for public URL

### "Permission Denied"

**Problem:** Camera/gallery permissions not granted

**Solution:**
- iOS: Settings → Expo Go → Enable Camera & Photos
- Android: Settings → Apps → Expo Go → Permissions

### "Node version warnings"

**Problem:** Node 16.x is too old for React Native 0.81

**Solution:**
```bash
# Install Node 20+
nvm install 20
nvm use 20

# Or use Homebrew
brew install node@20
```

### "Module not found"

**Problem:** Dependencies not installed

**Solution:**
```bash
rm -rf node_modules
npm install
```

## Building for Production

### iOS (requires Mac + Xcode)

```bash
# Build standalone app
npx eas build --platform ios

# Or for App Store
npx eas submit --platform ios
```

### Android

```bash
# Build APK
npx eas build --platform android --profile preview

# Or for Play Store
npx eas build --platform android
npx eas submit --platform android
```

You'll need to create an Expo account: `npx eas login`

## Known Issues

1. **Node 16.x**: Upgrade to Node 18+ or 20+ for best compatibility
2. **Localhost**: Won't work on physical devices - use local IP or ngrok
3. **Large images**: Automatically compressed to 80% quality before upload

## Supported Landmarks

- 🏰 Gediminas Tower
- ⛪ Vilnius Cathedral
- 🚪 Gate of Dawn
- 🏛️ St. Anne's Church
- ✝️ Three Crosses Monument

## Dependencies

- **React Native** - Mobile framework
- **Expo** - Development tools
- **React Navigation** - Screen navigation
- **Expo Camera** - Camera access
- **Expo Image Picker** - Gallery access
- **Axios** - HTTP client
- **React Native Paper** - UI components

## Development

### Hot Reload

Changes are automatically reloaded. Just save your file!

### Debugging

- Shake device → "Debug Remote JS"
- Or press `j` in terminal → Opens Chrome DevTools
- View console logs: `npx react-native log-ios` or `log-android`

### Adding New Features

1. **New Screen:**
   - Create file in `src/screens/`
   - Add to navigation in `App.tsx`

2. **New API Endpoint:**
   - Add to `src/services/api.ts`
   - Update types in `src/types/index.ts`

## Performance

- Image upload: ~2-5 seconds (depends on network)
- Prediction inference: ~150-500ms (backend)
- History load: <1 second

## License

See main project README.

## Support

- Backend API docs: `http://localhost:5126/swagger`
- React Native docs: https://reactnative.dev
- Expo docs: https://docs.expo.dev

---

**Ready to test!** 🚀

1. Start PostgreSQL
2. Start backend API
3. Update IP in config.ts
4. Run `npm start`
5. Scan QR code with Expo Go
