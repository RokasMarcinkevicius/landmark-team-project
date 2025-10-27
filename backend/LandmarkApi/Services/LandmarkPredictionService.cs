using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace LandmarkApi.Services;

public class LandmarkPrediction
{
    public required string Label { get; set; }
    public float Confidence { get; set; }
    public int Rank { get; set; }
}

public class PredictionResult
{
    public required List<LandmarkPrediction> Predictions { get; set; }
    public long InferenceTimeMs { get; set; }
}

/// <summary>
/// Landmark prediction service using Python TFLite interpreter via subprocess
/// Note: This is a bridge solution. For production, consider:
/// 1. Converting model to ONNX format and using Microsoft.ML.OnnxRuntime
/// 2. Using TensorFlow.NET (if stable for your platform)
/// 3. Creating a separate Python microservice
/// </summary>
public class LandmarkPredictionService : IDisposable
{
    private readonly string[] _labels;
    private readonly string _modelPath;
    private readonly string _pythonPath;
    private const int ImageSize = 224;
    private readonly ILogger<LandmarkPredictionService> _logger;

    public LandmarkPredictionService(ILogger<LandmarkPredictionService> logger, IConfiguration configuration)
    {
        _logger = logger;

        _modelPath = configuration["ModelPath"] ?? "Models/landmark_mnv3_int8_drq.tflite";
        var labelsPath = configuration["LabelsPath"] ?? "Models/labels.txt";
        _pythonPath = configuration["PythonPath"] ?? "/Users/rokas/Documents/KTU AI Masters/Team Project/.venv/bin/python";

        _logger.LogInformation($"Model path: {_modelPath}");
        _logger.LogInformation($"Labels path: {labelsPath}");

        if (!File.Exists(_modelPath))
            throw new FileNotFoundException($"Model file not found: {_modelPath}");

        if (!File.Exists(labelsPath))
            throw new FileNotFoundException($"Labels file not found: {labelsPath}");

        // Load labels
        _labels = File.ReadAllLines(labelsPath);

        _logger.LogInformation($"Service initialized. {_labels.Length} classes detected");
    }

    public async Task<PredictionResult> PredictAsync(Stream imageStream)
    {
        var stopwatch = Stopwatch.StartNew();

        // Save image temporarily
        var tempImagePath = Path.Combine(Path.GetTempPath(), $"temp_image_{Guid.NewGuid()}.jpg");
        try
        {
            // Load and preprocess image
            using (var image = await Image.LoadAsync<Rgb24>(imageStream))
            {
                // Resize to 224x224
                image.Mutate(x => x.Resize(ImageSize, ImageSize));
                await image.SaveAsJpegAsync(tempImagePath);
            }

            // Create Python script path
            var scriptPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "predict_tflite.py");

            // If script doesn't exist, use Python inference
            if (!File.Exists(scriptPath))
            {
                _logger.LogWarning("Python script not found. Creating it...");
                await CreatePythonScript(scriptPath);
            }

            // Call Python script
            var result = await RunPythonInference(tempImagePath);

            stopwatch.Stop();
            result.InferenceTimeMs = stopwatch.ElapsedMilliseconds;

            return result;
        }
        finally
        {
            // Clean up temp file
            if (File.Exists(tempImagePath))
            {
                File.Delete(tempImagePath);
            }
        }
    }

    private async Task<PredictionResult> RunPythonInference(string imagePath)
    {
        var scriptPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "predict_tflite.py");

        var psi = new ProcessStartInfo
        {
            FileName = _pythonPath,
            Arguments = $"\"{scriptPath}\" \"{_modelPath}\" \"{imagePath}\"",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        using var process = new Process { StartInfo = psi };
        process.Start();

        var output = await process.StandardOutput.ReadToEndAsync();
        var error = await process.StandardError.ReadToEndAsync();
        await process.WaitForExitAsync();

        if (process.ExitCode != 0)
        {
            throw new Exception($"Python inference failed: {error}");
        }

        // Parse JSON output from Python script
        var predictions = JsonSerializer.Deserialize<List<PythonPrediction>>(output)
            ?? throw new Exception("Failed to parse Python output");

        return new PredictionResult
        {
            Predictions = predictions.Select((p, idx) => new LandmarkPrediction
            {
                Label = _labels[p.ClassIndex],
                Confidence = p.Confidence,
                Rank = idx + 1
            }).ToList(),
            InferenceTimeMs = 0 // Will be set by caller
        };
    }

    private async Task CreatePythonScript(string scriptPath)
    {
        var pythonScript = @"
import sys
import json
import numpy as np
import tensorflow as tf
from PIL import Image

def predict(model_path, image_path):
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Load and preprocess image
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Run inference
    interpreter.set_tensor(input_details['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details['index'])[0]

    # Get top-3
    top3_idx = np.argsort(output)[-3:][::-1]
    result = [{'class_index': int(idx), 'confidence': float(output[idx])} for idx in top3_idx]

    print(json.dumps(result))

if __name__ == '__main__':
    predict(sys.argv[1], sys.argv[2])
";
        await File.WriteAllTextAsync(scriptPath, pythonScript);
        _logger.LogInformation($"Created Python inference script at {scriptPath}");
    }

    public void Dispose()
    {
        // Cleanup if needed
    }

    private class PythonPrediction
    {
        [JsonPropertyName("class_index")]
        public int ClassIndex { get; set; }

        [JsonPropertyName("confidence")]
        public float Confidence { get; set; }
    }
}
