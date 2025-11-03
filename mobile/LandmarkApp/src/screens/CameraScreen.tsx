// src/screens/CameraScreen.tsx
import React, { useState } from 'react';
import {
    View,
    StyleSheet,
    Text,
    Image,
    Alert,
    ActivityIndicator,       // <-- added (react-native)
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Button, Card } from 'react-native-paper';
import { predictLandmark } from '../services/api';
import { PredictionResult } from '../types';

interface CameraScreenProps {
    navigation: any;
}

export default function CameraScreen({ navigation }: CameraScreenProps) {
    const [selectedImage, setSelectedImage] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);

    // Request camera permissions and take photo
    const takePhoto = async () => {
        const { status } = await ImagePicker.requestCameraPermissionsAsync();

        if (status !== 'granted') {
            Alert.alert('Permission Denied', 'Camera permission is required to take photos.');
            return;
        }

        const result = await ImagePicker.launchCameraAsync({
            // use the documented enum constant
            mediaTypes: ImagePicker.MediaTypeOptions.Images,
            allowsEditing: true,
            aspect: [4, 3],
            quality: 0.8,
        });

        if (!result.canceled && result.assets?.[0]) {
            setSelectedImage(result.assets[0].uri);
        }
    };

    // Pick image from gallery
    const pickImage = async () => {
        const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();

        if (status !== 'granted') {
            Alert.alert('Permission Denied', 'Gallery permission is required to select photos.');
            return;
        }

        const result = await ImagePicker.launchImageLibraryAsync({
            mediaTypes: ImagePicker.MediaTypeOptions.Images,
            allowsEditing: true,
            aspect: [4, 3],
            quality: 0.8,
        });

        if (!result.canceled && result.assets?.[0]) {
            setSelectedImage(result.assets[0].uri);
        }
    };

    // Upload image and get prediction
    const handlePredict = async () => {
        if (!selectedImage) {
            Alert.alert('No Image', 'Please take a photo or select one from gallery.');
            return;
        }

        setLoading(true);

        try {
            const result: PredictionResult = await predictLandmark(selectedImage);

            navigation.navigate('Results', {
                prediction: result,
                imageUri: selectedImage,
            });
        } catch (error: any) {
            console.error('Prediction error:', error);
            Alert.alert(
                'Prediction Failed',
                error?.response?.data?.error ?? error?.message ?? 'Failed to connect to server.'
            );
        } finally {
            setLoading(false);
        }
    };

    return (
        <View style={styles.container}>
            <Text style={styles.title}>Vilnius Landmark Identifier</Text>
            <Text style={styles.subtitle}>Take a photo or select one from your gallery</Text>

            {selectedImage && (
                <Card style={styles.imageCard}>
                    <Image source={{ uri: selectedImage }} style={styles.image} />
                </Card>
            )}

            <View style={styles.buttonContainer}>
                <Button
                    mode="contained"
                    icon="camera"
                    onPress={takePhoto}
                    style={[styles.button, styles.buttonSpacing]}
                    disabled={loading}
                >
                    Take Photo
                </Button>

                <Button
                    mode="contained"
                    icon="image"
                    onPress={pickImage}
                    style={[styles.button, styles.buttonSpacing]}
                    disabled={loading}
                >
                    Choose from Gallery
                </Button>

                {selectedImage && (
                    <Button
                        mode="contained"
                        icon="send"
                        onPress={handlePredict}
                        style={[styles.button, styles.predictButton]}
                        disabled={loading}
                        loading={loading} // make sure loading is boolean (it is)
                    >
                        {loading ? 'Analyzing...' : 'Identify Landmark'}
                    </Button>
                )}
            </View>

            {loading && (
                <View style={styles.loadingOverlay}>
                    <ActivityIndicator size="large" color="#6200ea" />
                    <Text style={styles.loadingText}>Analyzing image...</Text>
                </View>
            )}
        </View>
    );
}

const styles = StyleSheet.create({
    container: { flex: 1, padding: 20, backgroundColor: '#f5f5f5' },
    title: { fontSize: 24, fontWeight: 'bold', textAlign: 'center', marginTop: 20, marginBottom: 10, color: '#333' },
    subtitle: { fontSize: 14, textAlign: 'center', marginBottom: 30, color: '#666' },
    imageCard: { marginBottom: 20, elevation: 4 },
    image: { width: '100%', height: 300, borderRadius: 8 },
    buttonContainer: { /* removed gap; keep column layout */ },
    button: { paddingVertical: 8 },
    buttonSpacing: { marginBottom: 12 },
    predictButton: { backgroundColor: '#6200ea', marginTop: 10 },
    loadingOverlay: {
        position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
        backgroundColor: 'rgba(0,0,0,0.7)', justifyContent: 'center', alignItems: 'center'
    },
    loadingText: { color: '#fff', marginTop: 10, fontSize: 16 },
});