import os
import csv
from datetime import datetime
from typing import Dict

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# -------------------------------------------------------------
# You MUST keep this in sync with your training architectures
# -------------------------------------------------------------
from torchvision.models import vgg16



# -------------------------------------------------------------
# Image preprocessing (same as training)
# -------------------------------------------------------------
def _get_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# -------------------------------------------------------------
# Build model architecture (must match training)
# -------------------------------------------------------------
def _build_model(model_key, num_classes):

    if model_key == "vgg16":

        model = vgg16(weights=None)

        model.classifier = nn.Sequential(
            nn.Linear(25088, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    return model


# -------------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------------
def run_logging_for_all_crops_and_splits(
    crop_datasets: Dict[str, Dict[str, str]],
    models: Dict[str, str],
    output_csv: str,
    img_size: int = 224,
    device: torch.device = None
):
    """
    crop_datasets = {
        "corn": {
            "validation": "/path/to/corn/val",
            "test": "/path/to/corn/test"
        },
        "wheat": {
            "validation": "/path/to/wheat/val",
            "test": "/path/to/wheat/test"
        }
    }

    models = {
        "efficientnet_b0": "/path/to/model.pth",
        "mobilenet_v2": "/path/to/model.pth"
    }
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = _get_transform(img_size)

    file_exists = os.path.isfile(output_csv)

    with open(output_csv, "a", newline="", encoding="utf-8") as f:

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

        # -------------------------------------------------
        # Loop crops
        # -------------------------------------------------
        for crop_name, split_map in crop_datasets.items():

            # ---------------------------------------------
            # Loop splits (validation / test)
            # ---------------------------------------------
            for split_name, data_dir in split_map.items():

                if not os.path.isdir(data_dir):
                    print(f"[WARN] Missing folder: {data_dir}")
                    continue

                class_names = sorted(
                    d for d in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir, d))
                )

                if len(class_names) == 0:
                    print(f"[WARN] No class folders in {data_dir}")
                    continue

                # -----------------------------------------
                # Loop models
                # -----------------------------------------
                for model_key, model_path in models.items():

                    if not os.path.isfile(model_path):
                        print(f"[WARN] Model not found: {model_path}")
                        continue

                    print(
                        f"Logging → Crop={crop_name} | "
                        f"Split={split_name} | "
                        f"Model={model_key}"
                    )

                    model = _build_model(
                        model_key=model_key,
                        num_classes=len(class_names)
                    )

                    checkpoint = torch.load(
                        model_path,
                        map_location=device
                    )

                    # handle both raw state_dict and wrapped dict
                    if isinstance(checkpoint, dict) and \
                       "state_dict" in checkpoint:
                        model.load_state_dict(checkpoint["state_dict"])
                    else:
                        model.load_state_dict(checkpoint)

                    model.to(device)
                    model.eval()

                    with torch.no_grad():

                        # ---------------------------------
                        # Loop classes
                        # ---------------------------------
                        for true_label in class_names:

                            class_dir = os.path.join(
                                data_dir,
                                true_label
                            )

                            for img_name in os.listdir(class_dir):

                                img_path = os.path.join(
                                    class_dir,
                                    img_name
                                )

                                if not os.path.isfile(img_path):
                                    continue

                                try:
                                    img = Image.open(
                                        img_path
                                    ).convert("RGB")

                                    x = transform(img) \
                                        .unsqueeze(0) \
                                        .to(device)

                                    logits = model(x)
                                    probs = torch.softmax(
                                        logits, dim=1
                                    )

                                    conf, pred_idx = torch.max(
                                        probs, dim=1
                                    )

                                    predicted_label = class_names[
                                        int(pred_idx.item())
                                    ]

                                    is_correct = int(
                                        predicted_label == true_label
                                    )

                                    writer.writerow([
                                        img_name,
                                        datetime.now().strftime("%Y-%m-%d"),
                                        crop_name,
                                        true_label,
                                        predicted_label,
                                        float(conf.item()),
                                        is_correct,
                                        split_name,
                                        model_key
                                    ])

                                except Exception as e:
                                    print(
                                        "[ERROR] Skipping:",
                                        img_path,
                                        str(e)
                                    )

    print("\nPrediction logging completed.")
