import os
import csv
from datetime import datetime

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# ---------------- CONFIG ----------------

current=os.path.dirname(os.path.abspath(__file__))
DATA_DIR=os.path.abspath(os.path.join(current,'..','data','corn'))


MODELS_DIR = os.path.abspath(os.path.join(current,'..','saved_models'))           # folder containing .pt / .pth models
#DATA_DIR   = r"D:\dataset\test"     # test / validation folder
IMG_SIZE   = 224
DATASET_TYPE = "test"               # or "validation"
OUT_CSV = "predictions_log.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------


# class names from folder structure
class_names = sorted([
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d))
])


# image transform (same as training ideally)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])


def load_model(model_path, num_classes):
    """
    IMPORTANT:
    Replace this function body with
    your actual model architecture.
    """

    # Example: if you used a ResNet-like model
    # from torchvision.models import resnet50
    # model = resnet50(pretrained=False)
    # model.fc = nn.Linear(model.fc.in_features, num_classes)

    # ------------- YOU MUST ADAPT THIS -------------
    raise NotImplementedError(
        "Define your model architecture here before loading weights."
    )


def predict_with_model(model_path):

    print("Loading model:", model_path)

    # ---- build model arch and load weights ----
    model = load_model(model_path, num_classes=len(class_names))

    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)

    model.to(DEVICE)
    model.eval()

    model_name = os.path.splitext(os.path.basename(model_path))[0]

    rows = []

    with torch.no_grad():

        for true_label in class_names:

            class_dir = os.path.join(DATA_DIR, true_label)
            crop = true_label.split("___")[0]

            true_index = class_names.index(true_label)

            for img_name in os.listdir(class_dir):

                img_path = os.path.join(class_dir, img_name)

                try:
                    img = Image.open(img_path).convert("RGB")
                    x = transform(img).unsqueeze(0).to(DEVICE)

                    outputs = model(x)
                    probs = torch.softmax(outputs, dim=1)

                    confidence, pred_index = torch.max(probs, dim=1)

                    predicted_label = class_names[int(pred_index.item())]
                    confidence = float(confidence.item())

                    is_correct = int(predicted_label == true_label)

                    rows.append([
                        img_name,
                        datetime.now().strftime("%Y-%m-%d"),
                        crop,
                        true_label,
                        predicted_label,
                        confidence,
                        is_correct,
                        DATASET_TYPE,
                        model_name
                    ])

                except Exception as e:
                    print("Skipped:", img_path, e)

    return rows


# ---------------- MAIN ----------------

all_rows = []

model_files = [
    os.path.join(MODELS_DIR, f)
    for f in os.listdir(MODELS_DIR)
    if f.endswith(".pt") or f.endswith(".pth")
]

for m in model_files:
    model_rows = predict_with_model(m)
    all_rows.extend(model_rows)


file_exists = os.path.isfile(OUT_CSV)

with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)

    if not file_exists:
        writer.writerow([
            "image_id",
            "date",
            "crop",
            "true_label",
            "predicted_label",
            "confidence",
            "is_correct",
            "dataset_type",
            "model_name"
        ])

    writer.writerows(all_rows)

print("Done. Total logged rows:", len(all_rows))
print("Saved to:", OUT_CSV)
