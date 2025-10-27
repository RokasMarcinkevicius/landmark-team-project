using Microsoft.EntityFrameworkCore;
using LandmarkApi.Models;

namespace LandmarkApi.Data;

public class LandmarkDbContext : DbContext
{
    public LandmarkDbContext(DbContextOptions<LandmarkDbContext> options) : base(options)
    {
    }

    public DbSet<PredictionRecord> PredictionRecords { get; set; }

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        base.OnModelCreating(modelBuilder);

        modelBuilder.Entity<PredictionRecord>(entity =>
        {
            entity.HasKey(e => e.Id);
            entity.Property(e => e.ImagePath).IsRequired().HasMaxLength(500);
            entity.Property(e => e.Timestamp).IsRequired();
            entity.HasIndex(e => e.Timestamp);
        });

        modelBuilder.Entity<PredictionDetail>(entity =>
        {
            entity.HasKey(e => e.Id);
            entity.HasOne(e => e.PredictionRecord)
                .WithMany(p => p.Predictions)
                .HasForeignKey(e => e.PredictionRecordId)
                .OnDelete(DeleteBehavior.Cascade);
        });
    }
}
