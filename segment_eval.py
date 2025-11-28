import os
from tkinter import image_names
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from CustomDataset import CustomDatasett
from models import get_model
from PIL import Image
import pytorch_lightning as pl
import torch.nn.functional as F
import cv2
import numpy as np
from train import *
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

def clean_segmentation(mask, min_area=10):
    """
    Removes small blobs using connected components.
    """
    mask_np = mask.cpu().numpy().astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
    cleaned = np.zeros_like(mask_np)

    for i in range(1, num_labels):  # skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return torch.tensor(cleaned, device=mask.device)

def keep_lowest_component(mask):
    """
    Keep only the connected component closest to the bottom of the image.
    """
    mask_np = mask.cpu().numpy().astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
    if num_labels <= 1:  # No components except background
        return mask

    # Find the label whose centroid has the largest y-coordinate (lowest)
    lowest_idx = 1 + np.argmax(centroids[1:, 1])  # Skip background
    filtered = np.where(labels == lowest_idx, 255, 0).astype(np.uint8)
    return torch.tensor(filtered, device=mask.device)


def apply_dilation(mask, kernel_size=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_np = mask.cpu().numpy().astype(np.uint8)
    dilated_mask_np = cv2.dilate(mask_np, kernel, iterations=2)
    return torch.tensor(dilated_mask_np, device=mask.device)


# Set the paths to the image and mask folders
mask_dir = r"C:\Users\jaoaa\Ambiente de Trabalho\Diogo_projects\MACTEL\sequential\All_masks"
image_dir = r"C:\Users\jaoaa\Ambiente de Trabalho\Diogo_projects\MACTEL\sequential\All"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def apply_custom_threshold(mask, threshold):
    return torch.where(mask <= threshold, torch.tensor(0, device=mask.device), torch.tensor(255, device=mask.device))

def apply_erosion(mask, kernel_size=1, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_np = mask.cpu().numpy().astype(np.uint8)
    eroded_mask_np = cv2.erode(mask_np, kernel, iterations=iterations)
    return torch.tensor(eroded_mask_np, device=mask.device)

def segment_images(model, img_folder, mask_folder):
    os.makedirs(mask_folder, exist_ok=True)

    transform = ToTensor()
    dataset = CustomDatasett(img_folder, img_folder, transform=transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"Number of images: {len(dataset)}")
    print(f"Number of batches: {len(data_loader)}")

    if len(data_loader) == 0:
        print("Data loader is empty.")


    
    with torch.no_grad():
        for i, (image, _, image_name) in enumerate(data_loader):
            image = image.to(device)
            mask = model(image)
            mask = torch.sigmoid(mask)
            mask = (mask > 0.5).float() * 255  # Binary thresholding
            # Apply erosion if needed
            # mask_2d = apply_erosion(mask_2d)

            # Remove small blobs

            # Apply dilation
            #mask = apply_dilation(mask[0, 0])
            mask = apply_erosion(mask[0, 0])
            mask_2d = clean_segmentation(mask)
            mask = keep_lowest_component(mask)
            image_name = image_name[0].split(".")[0] + "_mask.png"
            mask_path = os.path.join(mask_folder, image_name)
            #mask_image = Image.fromarray(mask_2d.byte().cpu().numpy(), mode="L")
            mask_image = Image.fromarray(mask_2d.byte().cpu().numpy())
            mask_image.save(mask_path)

    print("Segmentation complete.")

if __name__ == '__main__':
    state_dict=torch.load(r"C:\Users\jaoaa\Ambiente de Trabalho\Thesis\src\paths\drunet_pretrained_band2_5_dict.pth", map_location=device)
    consume_prefix_in_state_dict_if_present(state_dict, prefix="unet.")

    model = get_model("drunet")  # must match architecture
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    segment_images(model, image_dir, mask_dir)