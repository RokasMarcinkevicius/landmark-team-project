import wandb, os
run = wandb.init(project="landmark-id", job_type="dataset_upload")
artifact = wandb.Artifact("vilnius-landmarks-mini", type="dataset",
                          description="5-class Vilnius landmarks, Commons; split 70/15/15, seed=42")
artifact.add_dir("data/train", name="train")
artifact.add_dir("data/val", name="val")
artifact.add_dir("data/test", name="test")
artifact.add_file("data/SOURCES.csv")
wandb.log_artifact(artifact)
run.finish()
