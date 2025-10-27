#!/bin/bash
# Quick start script for Landmark App

echo "========================================="
echo "Landmark Identifier - React Native App"
echo "========================================="
echo ""

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
    echo ""
fi

# Get local IP
echo "🌐 Your local IP addresses:"
ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print "   " $2}'
echo ""
echo "⚠️  Update src/constants/config.ts with your IP address!"
echo ""

# Check if backend is running
echo "🔍 Checking backend API..."
if curl -s http://localhost:5126/api/prediction/health > /dev/null 2>&1; then
    echo "✅ Backend API is running"
else
    echo "❌ Backend API is NOT running"
    echo "   Start it with: cd ../backend/LandmarkApi && dotnet run"
fi
echo ""

# Start Expo
echo "🚀 Starting Expo development server..."
echo "   Scan the QR code with Expo Go app on your phone"
echo ""
npm start
