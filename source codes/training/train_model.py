from ultralytics import YOLO
import yaml
from pathlib import Path
import torch


class DefectDetectionTrainer:
    """Train YOLO model for smartphone defect detection"""

    def __init__(self, model_size='n'):
        """
        Initialize trainer

        Args:
            model_size: 'n'=nano, 's'=small, 'm'=medium, 'l'=large, 'x'=extra-large
                       Start with 'n' for faster training, upgrade later if needed
        """
        self.model_size = model_size
        self.model = None
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)

        # Check if GPU is available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f" Using device: {self.device}")
        if self.device == 'cpu':
            print(
                "⚠️ Training on CPU will be slow. Consider using Google Colab for GPU access.")

    def load_model(self):
        """Load pre-trained YOLO model"""

        model_path = f'yolov8{self.model_size}.pt'
        print(f" Loading YOLOv8{self.model_size} model...")

        try:
            self.model = YOLO(model_path)
            print("Model loaded successfully!")

            # Print model info
            self.model.info()

        except Exception as e:
            print(f"Error loading model: {e}")
            print("The model will be downloaded automatically on first use")

    def train(self, epochs=50, batch_size=16, img_size=640):
        """
        Train the defect detection model

        Args:
            epochs: Number of training iterations (start with 50)
            batch_size: Images processed at once (reduce if out of memory)
            img_size: Input image size (640 is good balance)
        """

        if self.model is None:
            self.load_model()

        # Dataset configuration file
        data_config = "configs/dataset.yaml"

        if not Path(data_config).exists():
            print(f"Dataset config not found: {data_config}")
            print("Make sure you've run the data processing step")
            return None

        print("Starting training...")
        print(f"Configuration:")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Image size: {img_size}x{img_size}")

        try:
            # Start training
            results = self.model.train(
                data=data_config,       # Dataset config
                epochs=epochs,          # Training epochs
                batch=batch_size,       # Batch size
                imgsz=img_size,        # Image size
                device=self.device,     # GPU/CPU
                patience=10,           # Early stopping patience
                save=True,             # Save checkpoints
                plots=True,            # Generate plots
                verbose=True           # Verbose output
            )

            print(" Training completed!")
            print(f" Results saved to: {results.save_dir}")

            return results

        except Exception as e:
            print(f"❌ Training failed: {e}")
            return None

    def validate_model(self):
        """Validate the trained model"""

        if self.model is None:
            print("❌ No model loaded")
            return

        print("Validating model...")

        try:
            # Run validation
            results = self.model.val()

            print(" Validation Results:")
            print(f"   mAP@0.5: {results.box.map50:.4f}")
            print(f"   mAP@0.5:0.95: {results.box.map:.4f}")

            return results

        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return None

# Example usage and training script


def main():
    """Main training pipeline"""

    print("Defect Detection Model Training")
    print("=" * 50)

    # Check dataset
    dataset_config = Path("configs/dataset.yaml")
    if not dataset_config.exists():
        print("❌ Dataset not ready. Please run data processing first.")
        return

    # Initialize trainer
    trainer = DefectDetectionTrainer(model_size='s')  # Start with small

    # Train model
    print("\nPhase 1: Training")
    results = trainer.train(
        epochs=50,      # Start small, increase later
        batch_size=8,  # Reduce to 8 if out of memory
        img_size=640
    )

    if results is None:
        print("❌ Training failed")
        return

    # Validate model
    print("\nPhase 2: Validation")
    val_results = trainer.validate_model()

    if val_results:
        print("\nTraining pipeline completed successfully!")
        print("   Check results in 'results' folder")
        print("   Test model on new images")
    else:
        print("❌ Validation failed")


if __name__ == "__main__":
    main()
