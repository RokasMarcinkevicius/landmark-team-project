using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace LandmarkApi.Models;

/// <summary>
/// Stores prediction record with image and results
/// </summary>
public class PredictionRecord
{
    [Key]
    public int Id { get; set; }

    /// <summary>
    /// Path to stored image file
    /// </summary>
    [Required]
    [MaxLength(500)]
    public required string ImagePath { get; set; }

    /// <summary>
    /// Original image filename
    /// </summary>
    [MaxLength(255)]
    public string? OriginalFilename { get; set; }

    /// <summary>
    /// Image file size in bytes
    /// </summary>
    public long ImageSizeBytes { get; set; }

    /// <summary>
    /// When the prediction was made
    /// </summary>
    [Required]
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Inference time in milliseconds
    /// </summary>
    public long InferenceTimeMs { get; set; }

    /// <summary>
    /// Top prediction label
    /// </summary>
    [MaxLength(100)]
    public string? TopPrediction { get; set; }

    /// <summary>
    /// Top prediction confidence (0-1)
    /// </summary>
    public float TopConfidence { get; set; }

    /// <summary>
    /// Detailed predictions (Top-3)
    /// </summary>
    public ICollection<PredictionDetail> Predictions { get; set; } = new List<PredictionDetail>();
}

/// <summary>
/// Individual prediction result (part of Top-3)
/// </summary>
public class PredictionDetail
{
    [Key]
    public int Id { get; set; }

    [Required]
    public int PredictionRecordId { get; set; }

    [ForeignKey(nameof(PredictionRecordId))]
    public PredictionRecord? PredictionRecord { get; set; }

    /// <summary>
    /// Landmark label
    /// </summary>
    [Required]
    [MaxLength(100)]
    public required string Label { get; set; }

    /// <summary>
    /// Confidence score (0-1)
    /// </summary>
    [Required]
    public float Confidence { get; set; }

    /// <summary>
    /// Rank (1, 2, or 3)
    /// </summary>
    [Required]
    public int Rank { get; set; }
}
