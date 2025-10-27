// TypeScript types for the Landmark Identification app

export interface LandmarkPrediction {
  label: string;
  confidence: number;
  rank: number;
}

export interface PredictionResult {
  predictions: LandmarkPrediction[];
  inferenceTimeMs: number;
}

export interface HistoryRecord {
  id: number;
  originalFilename: string;
  timestamp: string;
  topPrediction: string;
  topConfidence: number;
  inferenceTimeMs: number;
  imageSizeMb: number;
}

export const LANDMARK_NAMES: Record<string, string> = {
  gediminas_tower: 'Gediminas Tower',
  vilnius_cathedral: 'Vilnius Cathedral',
  gate_of_dawn: 'Gate of Dawn',
  st_anne: "St. Anne's Church",
  three_crosses: 'Three Crosses Monument',
};

export const CONFIDENCE_THRESHOLD = 0.55; // Below this is "unsure"
