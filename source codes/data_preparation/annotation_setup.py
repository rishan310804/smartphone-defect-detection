import json
import os
from pathlib import Path


class AnnotationManager:
    """Manages the annotation process for defect detection"""

    def __init__(self):
        self.image_dir = Path("data/raw/images")
        self.annotation_dir = Path("data/annotations")
        self.annotation_dir.mkdir(parents=True, exist_ok=True)

        # Defect classes from SSGD dataset
        self.classes = [
            'crack', 'broken', 'spot', 'scratch',
            'light-leakage', 'blot', 'broken-membrane'
        ]

    def start_annotation_session(self):
        """Guide user through annotation process"""

        print("🏷️ Starting Annotation Session")
        print("=" * 50)
        print("📚 DEFECT TYPES TO LOOK FOR:")

        for i, defect in enumerate(self.classes, 1):
            print(f"{i}. {defect}")

        print("\n📋 ANNOTATION GUIDELINES:")
        print("- Draw tight boxes around defects")
        print("- Use correct class names (exact spelling)")
        print("- If unsure, skip the image")
        print("- Save frequently!")

        print(f"\n📁 Images to annotate: {self.image_dir}")
        print(f"📁 Annotations will save to: {self.annotation_dir}")

        print("\n🚀 Starting LabelMe...")
        print("To start: run 'labelme' in command line")

        # Create config file for LabelMe
        self.create_labelme_config()

    def create_labelme_config(self):
        """Create configuration file for LabelMe"""
        config = {
            "auto_save": True,
            "labels": self.classes,
            "default_shape_color": [0, 255, 0],
            "shape_color": "auto"
        }

        config_path = Path("configs/labelme_config.yaml")
        config_path.parent.mkdir(exist_ok=True)

        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        print(f"✅ Created LabelMe config: {config_path}")


if __name__ == "__main__":
    manager = AnnotationManager()
    manager.start_annotation_session()
