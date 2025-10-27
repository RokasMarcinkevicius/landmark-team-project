import axios from 'axios';
import { API_ENDPOINTS } from '../constants/config';
import { PredictionResult, HistoryRecord } from '../types';

// Create axios instance with timeout
const apiClient = axios.create({
  timeout: 30000, // 30 seconds
});

/**
 * Upload an image to the prediction API
 * @param imageUri - Local file URI from camera/gallery
 * @returns Prediction result with Top-3 predictions
 */
export const predictLandmark = async (imageUri: string): Promise<PredictionResult> => {
  const formData = new FormData();

  // Prepare image for upload
  const filename = imageUri.split('/').pop() || 'photo.jpg';
  const match = /\.(\w+)$/.exec(filename);
  const type = match ? `image/${match[1]}` : 'image/jpeg';

  formData.append('imageFile', {
    uri: imageUri,
    name: filename,
    type,
  } as any);

  const response = await apiClient.post<PredictionResult>(
    API_ENDPOINTS.PREDICT,
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  );

  return response.data;
};

/**
 * Get prediction history from the server
 * @param limit - Number of records to fetch (default: 20)
 * @returns Array of history records
 */
export const getHistory = async (limit: number = 20): Promise<HistoryRecord[]> => {
  const response = await apiClient.get<HistoryRecord[]>(
    `${API_ENDPOINTS.HISTORY}?limit=${limit}`
  );

  return response.data;
};

/**
 * Check if the API is healthy
 * @returns true if API is reachable and healthy
 */
export const checkHealth = async (): Promise<boolean> => {
  try {
    const response = await apiClient.get(API_ENDPOINTS.HEALTH, {
      timeout: 5000,
    });
    return response.status === 200;
  } catch (error) {
    console.error('Health check failed:', error);
    return false;
  }
};
