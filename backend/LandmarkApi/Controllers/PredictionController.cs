using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using LandmarkApi.Services;
using LandmarkApi.Data;
using LandmarkApi.Models;

namespace LandmarkApi.Controllers;

[ApiController]
[Route("api/[controller]")]
public class PredictionController : ControllerBase
{
    private readonly LandmarkPredictionService _predictionService;
    private readonly LandmarkDbContext _dbContext;
    private readonly ILogger<PredictionController> _logger;
    private readonly IWebHostEnvironment _environment;

    public PredictionController(
        LandmarkPredictionService predictionService,
        LandmarkDbContext dbContext,
        IWebHostEnvironment environment,
        ILogger<PredictionController> logger)
    {
        _predictionService = predictionService;
        _dbContext = dbContext;
        _environment = environment;
        _logger = logger;
    }

    /// <summary>
    /// Predict landmark from uploaded image
    /// </summary>
    /// <param name="imageFile">Image file (JPG, PNG)</param>
    /// <returns>Top-3 landmark predictions with confidence scores</returns>
    [HttpPost("predict")]
    [ProducesResponseType(typeof(PredictionResult), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    [ProducesResponseType(StatusCodes.Status500InternalServerError)]
    public async Task<ActionResult<PredictionResult>> Predict(IFormFile imageFile)
    {
        if (imageFile == null || imageFile.Length == 0)
        {
            return BadRequest(new { error = "No image file provided" });
        }

        // Validate file type
        var allowedExtensions = new[] { ".jpg", ".jpeg", ".png" };
        var extension = Path.GetExtension(imageFile.FileName).ToLowerInvariant();
        if (!allowedExtensions.Contains(extension))
        {
            return BadRequest(new { error = "Invalid file type. Only JPG and PNG are supported." });
        }

        // Validate file size (max 10MB)
        if (imageFile.Length > 10 * 1024 * 1024)
        {
            return BadRequest(new { error = "File size exceeds 10MB limit" });
        }

        try
        {
            _logger.LogInformation($"Predicting landmark for image: {imageFile.FileName} ({imageFile.Length} bytes)");

            // Save image to disk
            var uploadsFolder = Path.Combine(_environment.ContentRootPath, "uploads");
            Directory.CreateDirectory(uploadsFolder);

            var uniqueFileName = $"{Guid.NewGuid()}{extension}";
            var imagePath = Path.Combine(uploadsFolder, uniqueFileName);

            using (var fileStream = new FileStream(imagePath, FileMode.Create))
            {
                await imageFile.CopyToAsync(fileStream);
            }

            // Run prediction
            using var stream = imageFile.OpenReadStream();
            var result = await _predictionService.PredictAsync(stream);

            // Save to database
            var record = new PredictionRecord
            {
                ImagePath = imagePath,
                OriginalFilename = imageFile.FileName,
                ImageSizeBytes = imageFile.Length,
                Timestamp = DateTime.UtcNow,
                InferenceTimeMs = result.InferenceTimeMs,
                TopPrediction = result.Predictions[0].Label,
                TopConfidence = result.Predictions[0].Confidence,
                Predictions = result.Predictions.Select(p => new PredictionDetail
                {
                    Label = p.Label,
                    Confidence = p.Confidence,
                    Rank = p.Rank
                }).ToList()
            };

            _dbContext.PredictionRecords.Add(record);
            await _dbContext.SaveChangesAsync();

            _logger.LogInformation($"Prediction completed in {result.InferenceTimeMs}ms. Top prediction: {result.Predictions[0].Label} ({result.Predictions[0].Confidence:P2}). Saved to DB with ID: {record.Id}");

            return Ok(result);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during prediction");
            return StatusCode(500, new { error = "An error occurred during prediction", details = ex.Message });
        }
    }

    /// <summary>
    /// Get prediction history
    /// </summary>
    /// <param name="limit">Number of recent predictions to retrieve</param>
    [HttpGet("history")]
    public async Task<ActionResult> GetHistory([FromQuery] int limit = 20)
    {
        var records = await _dbContext.PredictionRecords
            .OrderByDescending(p => p.Timestamp)
            .Take(limit)
            .Select(p => new
            {
                p.Id,
                p.OriginalFilename,
                p.Timestamp,
                p.TopPrediction,
                p.TopConfidence,
                p.InferenceTimeMs,
                ImageSizeMb = p.ImageSizeBytes / (1024.0 * 1024.0)
            })
            .ToListAsync();

        return Ok(records);
    }

    /// <summary>
    /// Health check endpoint
    /// </summary>
    [HttpGet("health")]
    public IActionResult Health()
    {
        return Ok(new { status = "healthy", timestamp = DateTime.UtcNow });
    }
}
