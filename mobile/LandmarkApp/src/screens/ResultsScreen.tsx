import React from 'react';
import {
  View,
  StyleSheet,
  Text,
  Image,
  ScrollView,
} from 'react-native';
import { Card, Button, ProgressBar } from 'react-native-paper';
import { LANDMARK_NAMES, CONFIDENCE_THRESHOLD } from '../types';

interface ResultsScreenProps {
  route: any;
  navigation: any;
}

export default function ResultsScreen({ route, navigation }: ResultsScreenProps) {
  const { prediction, imageUri } = route.params;
  const topPrediction = prediction.predictions[0];
  const isConfident = topPrediction.confidence >= CONFIDENCE_THRESHOLD;

  const getDisplayName = (label: string) => {
    return LANDMARK_NAMES[label] || label;
  };

  const formatConfidence = (confidence: number) => {
    return `${(confidence * 100).toFixed(1)}%`;
  };

  return (
    <ScrollView style={styles.container}>
      {/* Image */}
      <Card style={styles.imageCard}>
        <Image source={{ uri: imageUri }} style={styles.image} />
      </Card>

      {/* Confidence warning */}
      {!isConfident && (
        <Card style={styles.warningCard}>
          <Card.Content>
            <Text style={styles.warningTitle}>⚠️ Low Confidence</Text>
            <Text style={styles.warningText}>
              The model is not very confident about this prediction.
              Try taking another photo with better lighting or angle.
            </Text>
          </Card.Content>
        </Card>
      )}

      {/* Top prediction */}
      <Card style={styles.topPredictionCard}>
        <Card.Content>
          <Text style={styles.topLabel}>Top Prediction</Text>
          <Text style={styles.topLandmark}>
            {getDisplayName(topPrediction.label)}
          </Text>
          <Text style={styles.topConfidence}>
            {formatConfidence(topPrediction.confidence)} confident
          </Text>
        </Card.Content>
      </Card>

      {/* All predictions */}
      <Card style={styles.predictionsCard}>
        <Card.Content>
          <Text style={styles.sectionTitle}>Top 3 Predictions</Text>

          {prediction.predictions.map((pred: any, index: number) => (
            <View key={index} style={styles.predictionItem}>
              <View style={styles.predictionHeader}>
                <Text style={styles.predictionRank}>#{pred.rank}</Text>
                <Text style={styles.predictionLabel}>
                  {getDisplayName(pred.label)}
                </Text>
                <Text style={styles.predictionConfidence}>
                  {formatConfidence(pred.confidence)}
                </Text>
              </View>
              <ProgressBar
                progress={pred.confidence}
                color={pred.rank === 1 ? '#6200ea' : '#9e9e9e'}
                style={styles.progressBar}
              />
            </View>
          ))}
        </Card.Content>
      </Card>

      {/* Inference time */}
      <Card style={styles.infoCard}>
        <Card.Content>
          <Text style={styles.infoText}>
            ⚡ Inference time: {prediction.inferenceTimeMs}ms
          </Text>
        </Card.Content>
      </Card>

      {/* Actions */}
      <View style={styles.buttonContainer}>
        <Button
          mode="contained"
          onPress={() => navigation.navigate('Camera')}
          style={styles.button}
        >
          Take Another Photo
        </Button>

        <Button
          mode="outlined"
          onPress={() => navigation.navigate('History')}
          style={styles.button}
        >
          View History
        </Button>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  imageCard: {
    margin: 15,
    elevation: 4,
  },
  image: {
    width: '100%',
    height: 250,
    borderRadius: 8,
  },
  warningCard: {
    margin: 15,
    marginTop: 0,
    backgroundColor: '#fff3cd',
    elevation: 2,
  },
  warningTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#856404',
    marginBottom: 5,
  },
  warningText: {
    fontSize: 14,
    color: '#856404',
  },
  topPredictionCard: {
    margin: 15,
    marginTop: 0,
    backgroundColor: '#6200ea',
    elevation: 4,
  },
  topLabel: {
    fontSize: 14,
    color: '#fff',
    opacity: 0.9,
    marginBottom: 5,
  },
  topLandmark: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 5,
  },
  topConfidence: {
    fontSize: 16,
    color: '#fff',
    opacity: 0.9,
  },
  predictionsCard: {
    margin: 15,
    marginTop: 0,
    elevation: 2,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 15,
    color: '#333',
  },
  predictionItem: {
    marginBottom: 20,
  },
  predictionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  predictionRank: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#6200ea',
    marginRight: 10,
  },
  predictionLabel: {
    flex: 1,
    fontSize: 16,
    color: '#333',
  },
  predictionConfidence: {
    fontSize: 14,
    color: '#666',
    fontWeight: '600',
  },
  progressBar: {
    height: 8,
    borderRadius: 4,
  },
  infoCard: {
    margin: 15,
    marginTop: 0,
    elevation: 1,
  },
  infoText: {
    fontSize: 14,
    color: '#666',
  },
  buttonContainer: {
    margin: 15,
    marginTop: 0,
    gap: 10,
    marginBottom: 30,
  },
  button: {
    paddingVertical: 8,
  },
});
