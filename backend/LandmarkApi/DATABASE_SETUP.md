# Database Setup Guide

## Quick Start with Docker (Recommended)

The easiest way to run PostgreSQL for development:

```bash
# Start PostgreSQL container
docker run --name landmark-postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=landmark_db \
  -p 5432:5432 \
  -d postgres:15

# Verify it's running
docker ps

# View logs
docker logs landmark-postgres
```

## Start the API

```bash
cd backend/LandmarkApi
dotnet run
```

The database will be automatically created and migrated on startup!

## Test the API with Database

```bash
# Upload an image
curl -X POST http://localhost:5126/api/prediction/predict \
  -F "imageFile=@/path/to/image.jpg"

# View history
curl http://localhost:5126/api/prediction/history

# View Swagger docs
open http://localhost:5126/swagger
```

## PostgreSQL Management

### Connect to database:
```bash
docker exec -it landmark-postgres psql -U postgres -d landmark_db
```

### Useful SQL queries:
```sql
-- View all predictions
SELECT * FROM "PredictionRecords" ORDER BY "Timestamp" DESC LIMIT 10;

-- Count predictions by landmark
SELECT "TopPrediction", COUNT(*)
FROM "PredictionRecords"
GROUP BY "TopPrediction";

-- View prediction details
SELECT pr."Id", pr."TopPrediction", pr."TopConfidence", pd."Label", pd."Confidence", pd."Rank"
FROM "PredictionRecords" pr
JOIN "PredictionDetail" pd ON pr."Id" = pd."PredictionRecordId"
ORDER BY pr."Timestamp" DESC;
```

### Stop/Start container:
```bash
docker stop landmark-postgres
docker start landmark-postgres
```

### Remove container:
```bash
docker stop landmark-postgres
docker rm landmark-postgres
```

## Alternative: Local PostgreSQL Installation

### macOS (Homebrew):
```bash
brew install postgresql@15
brew services start postgresql@15
createdb landmark_db
```

### Linux (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo -u postgres createdb landmark_db
```

### Windows:
Download from: https://www.postgresql.org/download/windows/

## Connection String

Update `appsettings.json` if needed:

```json
{
  "ConnectionStrings": {
    "DefaultConnection": "Host=localhost;Database=landmark_db;Username=postgres;Password=postgres"
  }
}
```

## Troubleshooting

### "Connection refused"
- Check if PostgreSQL is running: `docker ps` or `brew services list`
- Check port 5432 is available: `lsof -i :5432`

### "Password authentication failed"
- Verify password in connection string matches PostgreSQL password
- For Docker, password is set with `-e POSTGRES_PASSWORD=postgres`

### "Database does not exist"
- Database is auto-created by Entity Framework migrations
- Or create manually: `docker exec -it landmark-postgres createdb -U postgres landmark_db`

## Database Schema

The API automatically creates these tables:

**PredictionRecords** - Main prediction records
- Id (PK)
- ImagePath
- OriginalFilename
- ImageSizeBytes
- Timestamp
- InferenceTimeMs
- TopPrediction
- TopConfidence

**PredictionDetail** - Top-3 predictions per record
- Id (PK)
- PredictionRecordId (FK)
- Label
- Confidence
- Rank

## Production Deployment

For production, use managed PostgreSQL:
- **Heroku Postgres** (free tier)
- **AWS RDS**
- **DigitalOcean Managed Databases**
- **Azure Database for PostgreSQL**
- **Supabase** (includes API)

Update connection string in environment variables or `appsettings.Production.json`.
