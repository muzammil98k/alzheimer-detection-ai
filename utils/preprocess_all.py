import os
import nibabel as nib
import torch
import torch.nn.functional as F
from tqdm import tqdm

RAW_DIR = "data/raw"
OUTPUT_DIR = "data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def normalize(volume):
    volume = (volume - volume.min()) / (volume.max() - volume.min())
    return volume


def resize_volume(volume, size=(64, 64, 64)):

    volume = torch.tensor(volume)

    # add batch and channel dimensions
    volume = volume.unsqueeze(0).unsqueeze(0).float()

    volume = F.interpolate(
        volume,
        size=size,
        mode="trilinear",
        align_corners=False
    )

    return volume.squeeze()


for disc in os.listdir(RAW_DIR):

    disc_path = os.path.join(RAW_DIR, disc)

    if not os.path.isdir(disc_path):
        continue

    subjects = os.listdir(disc_path)

    for subject in tqdm(subjects, desc=f"Processing {disc}"):

        raw_path = os.path.join(disc_path, subject, "RAW")

        if not os.path.exists(raw_path):
            continue

        for file in os.listdir(raw_path):

            if "mpr-1_anon.img" in file:

                img_path = os.path.join(raw_path, file)

                try:

                    img = nib.load(img_path)
                    volume = img.get_fdata()

                    # remove extra dimension if exists
                    if volume.ndim == 4:
                        volume = volume[:, :, :, 0]

                    volume = normalize(volume)
                    volume = resize_volume(volume)

                    save_path = os.path.join(OUTPUT_DIR, subject + ".pt")

                    torch.save(volume, save_path)

                except Exception as e:

                    print(f"Error processing {img_path}: {e}")

print("\nPreprocessing completed successfully.")
