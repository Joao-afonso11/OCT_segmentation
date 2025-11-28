import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch


def binarize(image_array, threshold=127):
    return (image_array > threshold).astype(np.uint8)

def evaluate_segmentation(pred_folder, gt_folder):
    pred_files = sorted([f for f in os.listdir(pred_folder) if f.endswith(('.png', '.jpg'))])
    gt_files = sorted([f for f in os.listdir(gt_folder) if f.endswith(('.png', '.jpg'))])

    assert pred_files == gt_files, "Mismatch between predicted and GT filenames."

    total_pixels = 0
    correct_pixels = 0
    total_dice = 0
    total_precision = 0
    total_recall = 0
    total_pixel_diff = 0
    num_images = 0

    for filename in tqdm(pred_files, desc="Evaluating"):
        pred_img = np.array(Image.open(os.path.join(pred_folder, filename)).convert("L"))
        gt_img = np.array(Image.open(os.path.join(gt_folder, filename)).convert("L"))

        assert pred_img.shape == gt_img.shape, f"Image size mismatch in {filename}"

        pred_bin = binarize(pred_img)
        gt_bin = binarize(gt_img)

        # Pixel accuracy
        correct_pixels += np.sum(pred_bin == gt_bin)
        total_pixels += pred_bin.size

        # Dice
        intersection = np.logical_and(pred_bin, gt_bin).sum()
        union = pred_bin.sum() + gt_bin.sum()
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        total_dice += dice

        # Precision and Recall
        tp = np.logical_and(pred_bin == 1, gt_bin == 1).sum()
        fp = np.logical_and(pred_bin == 1, gt_bin == 0).sum()
        fn = np.logical_and(pred_bin == 0, gt_bin == 1).sum()

        precision = (tp + 1e-6) / (tp + fp + 1e-6)
        recall = (tp + 1e-6) / (tp + fn + 1e-6)
        total_precision += precision
        total_recall += recall

        # Total absolute pixel-wise difference
        abs_diff = np.abs(pred_bin.astype(np.float32) - gt_bin.astype(np.float32))
        total_pixel_diff += abs_diff.sum()
        print(filename)
        print(dice)
        print(total_pixel_diff)
        num_images += 1

    pixel_acc = correct_pixels / total_pixels
    avg_dice = total_dice / num_images
    avg_precision = total_precision / num_images
    avg_recall = total_recall / num_images
    avg_pixel_diff = total_pixel_diff / (num_images*256)

    print(f"Pixel Accuracy:        {pixel_acc:.4f}")
    print(f"Dice Score:            {avg_dice:.4f}")
    print(f"Precision:             {avg_precision:.4f}")
    print(f"Recall:                {avg_recall:.4f}")
    print(f"Average Pixel Diff:    {avg_pixel_diff:.4f}")

    return pixel_acc, avg_dice, avg_precision, avg_recall, avg_pixel_diff


evaluate_segmentation(r"C:\Users\jaoaa\Ambiente de Trabalho\Thesis\src\segmentations\All", r"C:\Users\jaoaa\Ambiente de Trabalho\Thesis\src\segmentations\Band 2")
