using Microsoft.AspNetCore.Mvc;
using LandmarkApi.Services;

namespace LandmarkApi.Controllers;

[ApiController]
[Route("api/[controller]")]
public class PredictionController : ControllerBase
{
    private readonly LandmarkPredictionService _predictionService;
    private readonly ILogger<PredictionController> _logger;

    public PredictionController(
        LandmarkPredictionService predictionService,
        ILogger<PredictionController> logger)
    {
        _predictionService = predictionService;
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

            using var stream = imageFile.OpenReadStream();
            var result = await _predictionService.PredictAsync(stream);

            _logger.LogInformation($"Prediction completed in {result.InferenceTimeMs}ms. Top prediction: {result.Predictions[0].Label} ({result.Predictions[0].Confidence:P2})");

            return Ok(result);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during prediction");
            return StatusCode(500, new { error = "An error occurred during prediction", details = ex.Message });
        }
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
