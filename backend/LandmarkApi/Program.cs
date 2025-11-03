using LandmarkApi.Services;
using LandmarkApi.Data;
using Microsoft.EntityFrameworkCore;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container
builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen(c =>
{
    c.SwaggerDoc("v1", new() {
        Title = "Landmark Identification API",
        Version = "v1",
        Description = "REST API for Vilnius landmark identification using TensorFlow Lite with PostgreSQL storage"
    });
});

// Configure PostgreSQL database
var connectionString = builder.Configuration.GetConnectionString("DefaultConnection")
    ?? "Host=localhost;Database=landmark_db;Username=postgres;Password=postgres";
builder.Services.AddDbContext<LandmarkDbContext>(options =>
    options.UseNpgsql(connectionString));

// Register prediction service as singleton (model loaded once)
builder.Services.AddSingleton<LandmarkPredictionService>();

// Configure CORS for mobile app access
builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowAll", policy =>
    {
        policy.AllowAnyOrigin()
              .AllowAnyMethod()
              .AllowAnyHeader()
              .AllowAnyOrigin();
    });
});

// Configure file upload limits (10MB max)
builder.Services.Configure<Microsoft.AspNetCore.Http.Features.FormOptions>(options =>
{
    options.MultipartBodyLengthLimit = 10 * 1024 * 1024; // 10MB
});

var app = builder.Build();

// Auto-migrate database on startup
using (var scope = app.Services.CreateScope())
{
    var db = scope.ServiceProvider.GetRequiredService<LandmarkDbContext>();
    db.Database.Migrate();
    app.Logger.LogInformation("Database migrated successfully");
}

// Configure the HTTP request pipeline
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI(c =>
    {
        c.SwaggerEndpoint("/swagger/v1/swagger.json", "Landmark API v1");
    });
}

app.UseHttpsRedirection();
app.UseCors("AllowAll");
app.UseAuthorization();
app.MapControllers();

app.Logger.LogInformation("Landmark Identification API started");
app.Logger.LogInformation("Swagger UI available at: /swagger");
app.Urls.Add("http://0.0.0.0:5126");

app.Run();
