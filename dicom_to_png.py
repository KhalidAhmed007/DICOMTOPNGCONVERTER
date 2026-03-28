import os
import pydicom
import numpy as np
import cv2


INPUT_DIR = "data/raw_dicom"
OUTPUT_DIR = "data/processed_png"


def normalize(img):
    img = img.astype(np.float32)
    
    if np.max(img) == np.min(img):
        return np.zeros(img.shape, dtype=np.uint8)
    
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = (img * 255).astype(np.uint8)
    return img


def convert_all():
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith(".dcm"):
                
                dicom_path = os.path.join(root, file)

                # keep same folder structure
                rel_path = os.path.relpath(root, INPUT_DIR)
                save_dir = os.path.join(OUTPUT_DIR, rel_path)
                os.makedirs(save_dir, exist_ok=True)

                output_path = os.path.join(
                    save_dir, file.replace(".dcm", ".png")
                )

                try:
                    ds = pydicom.dcmread(dicom_path)
                    img = ds.pixel_array

                    # apply scaling if exists
                    if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
                        img = img * ds.RescaleSlope + ds.RescaleIntercept

                    img = normalize(img)

                    cv2.imwrite(output_path, img)

                except Exception as e:
                    print(f"Error: {dicom_path} -> {e}")


if __name__ == "__main__":
    convert_all()
    