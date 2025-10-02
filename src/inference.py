
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


class MaterialClassifier:

    
    def __init__(self, model_path="../models/resnet18_materials.pt", 
                 metadata_path="../models/model_metadata.json"):
        # Check files exist
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not Path(metadata_path).exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.class_names = self.metadata["class_names"]
        self.image_size = self.metadata["image_size"]
        
        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model on {self.device}...")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        
        # Preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])
        print("✓ Model loaded successfully\n")
    
    @torch.no_grad()
    def predict(self, image_path):
        """
        Predict material class for a single image
        
        Returns:
            pred_class (str): Predicted class name
            confidence (float): Confidence score (0-1)
            all_probs (dict): All class probabilities
        """
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            x = self.transform(img).unsqueeze(0).to(self.device)
            
            # Inference
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(logits.argmax(dim=1).cpu().numpy()[0])
            
            # Format results
            pred_class = self.class_names[pred_idx]
            confidence = float(probs[pred_idx])
            all_probs = {cls: float(prob) for cls, prob in zip(self.class_names, probs)}
            
            return pred_class, confidence, all_probs
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Material Classification Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py --image test.jpg
  python inference.py --image test.jpg --show-all
  python inference.py --image test.jpg --threshold 0.8
        """
    )
    
    parser.add_argument('--image', type=str, required=True, 
                       help='Path to input image')
    parser.add_argument('--model', type=str, default='../models/resnet18_materials.pt',
                       help='Path to TorchScript model')
    parser.add_argument('--metadata', type=str, default='../models/model_metadata.json',
                       help='Path to model metadata JSON')
    parser.add_argument('--show-all', action='store_true', 
                       help='Show all class probabilities')
    parser.add_argument('--threshold', type=float, default=0.0,
                       help='Confidence threshold (0-1)')
    
    args = parser.parse_args()
    
    # Check image exists
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    
    try:
        # Initialize classifier
        classifier = MaterialClassifier(
            model_path=args.model,
            metadata_path=args.metadata
        )
        
        # Predict
        pred_class, confidence, all_probs = classifier.predict(args.image)
        
        # Display results
        print("=" * 60)
        print(f"Image: {args.image}")
        print("=" * 60)
        print(f"Predicted Class: {pred_class}")
        print(f"Confidence:      {confidence:.2%}")
        
        # Check threshold
        if confidence < args.threshold:
            print(f"\n⚠️  Warning: Confidence below threshold ({args.threshold:.2%})")
        
        # Show all probabilities if requested
        if args.show_all:
            print(f"\nAll Class Probabilities:")
            sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
            for cls, prob in sorted_probs:
                bar = "█" * int(prob * 30)
                print(f"  {cls:15s}: {prob:6.2%} {bar}")
        
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()