import os
import csv
import torch
import torchvision.transforms as transforms
from PIL import Image
import shutil
import cv2
import numpy as np

# ---------------------------
# CONFIG
# ---------------------------
MODEL_PATH = "../models/resnet18_materials.pt"   # TorchScript model
DATA_FOLDER = "../Data/conveyor_frames"    # folder with images & videos
CSV_OUTPUT = "../results/conveyor_results.csv"
RETRAIN_QUEUE = "../Data/retrain_queue"
CONF_THRESHOLD = 0.70
CLASS_NAMES = ["e-waste", "fabric", "metal", "paper", "plastic"]

# ---------------------------
# Load model
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")
model = torch.jit.load(MODEL_PATH, map_location=device)
model.eval()
torch.backends.cudnn.benchmark = True

# ---------------------------
# Transforms
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------------------
# Predict helper
# ---------------------------
def predict_image(img):
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
    return probs.cpu().numpy()  # return full probability vector

# ---------------------------
# Manual override handler
# ---------------------------
def handle_override(user_input, file_name, pred_class):
    override_flag, correct_label = "NO", ""
    if user_input and user_input != "n":
        override_flag = "YES"
        if user_input == "y":
            correct_label = input(f"Enter correct class for {file_name}: ").strip().lower()
        elif user_input in CLASS_NAMES:
            correct_label = user_input

        if correct_label in CLASS_NAMES:
            save_path = os.path.join(RETRAIN_QUEUE, correct_label)
            os.makedirs(save_path, exist_ok=True)
            print(f" → Saved {file_name} to retrain queue under {correct_label}/")
    return override_flag, correct_label

# ---------------------------
# Conveyor simulation
# ---------------------------
def conveyor_simulation():
    files = sorted(os.listdir(DATA_FOLDER))
    os.makedirs(RETRAIN_QUEUE, exist_ok=True)

    with open(CSV_OUTPUT, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["File_Name", "Prediction", "Confidence", "Override_Flag", "Correct_Label"])

        for file_name in files:
            file_path = os.path.join(DATA_FOLDER, file_name)
            ext = file_name.lower().split('.')[-1]

            # --- Image ---
            if ext in ['png', 'jpg', 'jpeg']:
                img = Image.open(file_path).convert("RGB")
                probs = predict_image(img)
                pred_idx = probs.argmax()
                pred_class = CLASS_NAMES[pred_idx]
                conf = probs[pred_idx]
                print(f"[Image] {file_name} → Predicted: {pred_class}, Confidence: {conf:.2f}")
                if conf < CONF_THRESHOLD:
                    print(" ⚠️ Low confidence prediction!")
                user_input = input("Override? (y/n or new label): ").strip().lower()
                override_flag, correct_label = handle_override(user_input, file_name, pred_class)
                writer.writerow([file_name, pred_class, f"{conf:.2f}", override_flag, correct_label])

            # --- Video ---
            elif ext in ['mp4', 'avi', 'mov']:
                cap = cv2.VideoCapture(file_path)
                frame_probs = []

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img)
                    probs = predict_image(pil_img)
                    frame_probs.append(probs)

                cap.release()
                # Average probabilities across all frames
                avg_probs = np.mean(frame_probs, axis=0)
                pred_idx = avg_probs.argmax()
                pred_class = CLASS_NAMES[pred_idx]
                conf = avg_probs[pred_idx]
                print(f"[Video] {file_name} → Predicted: {pred_class}, Confidence: {conf:.2f}")
                if conf < CONF_THRESHOLD:
                    print(" ⚠️ Low confidence prediction!")
                user_input = input("Override? (y/n or new label): ").strip().lower()
                override_flag, correct_label = handle_override(user_input, file_name, pred_class)
                writer.writerow([file_name, pred_class, f"{conf:.2f}", override_flag, correct_label])


if __name__ == "__main__":
    conveyor_simulation()
