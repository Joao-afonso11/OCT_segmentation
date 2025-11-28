# train_mae.py

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from masked_autoencoder import MAESparK
from Dataset import OCTMaskedDataset
from visualize import save_recon_images
from tqdm import tqdm
import matplotlib.pyplot as plt


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def train():
    config = load_config("Deep_learning_approach\code\AE\config\config.yaml")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset = OCTMaskedDataset(
        root_dir=config["data"]["path"],
        patch_size=config["model"]["patch_size"],
        masking_ratio=config["model"]["masking_ratio"]
    )
    dataloader = DataLoader(dataset, batch_size=config["train"]["batch_size"], shuffle=True, num_workers=4)

    # Model
    model = MAESparK(
        encoder_cfg=config["model"]["encoder_cfg"],
        input_size=config["model"]["input_size"],
        patch_size=config["model"]["patch_size"], 
        embed_dim=config["model"]["embed_dim"],
        masking_ratio=config["model"]["masking_ratio"]
    ).to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=float(config["train"]["lr"]))
    criterion = nn.MSELoss()

    # Output dirs
    os.makedirs(config["train"]["output_dir"], exist_ok=True)
    os.makedirs(os.path.join(config["train"]["output_dir"], "recon"), exist_ok=True)
    loss_history = []

    # Training loop
    for epoch in range(config["train"]["epochs"]):
        model.train()
        epoch_loss = 0.0
        for images, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['train']['epochs']}"):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, masks) 

            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.6f}")
        plt.figure()
        plt.plot(loss_history, label="Total Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(config["train"]["output_dir"], "loss_curve.png"))
        plt.close()
        # Save checkpoint
        if (epoch + 1) % config["train"]["save_every"] == 0:
            ckpt_path = os.path.join(config["train"]["output_dir"], f"mae_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)

        # Visualization
        if (epoch + 1) % config["train"]["viz_every"] == 0:
            print("Before concatenation:")
            print(f"images: {images.shape}, outputs: {outputs.shape}, masks: {masks.shape}")
            save_recon_images(images, outputs, masks, epoch + 1, config["train"]["output_dir"])
            # Plot first sample's patch mask
            patch_h = config["model"]["input_size"] // config["model"]["patch_size"]
            patch_w = patch_h  # assuming square images

            mask_sample = masks[0].cpu().numpy().reshape(patch_h, patch_w)

            plt.figure(figsize=(4, 4))
            plt.imshow(mask_sample, cmap="gray")
            plt.title(f"Mask (Epoch {epoch + 1})")
            plt.axis("off")
            mask_path = os.path.join(config["train"]["output_dir"], f"mask_epoch_{epoch + 1}.png")
            plt.savefig(mask_path)
            plt.close()
            print(f"[✓] Saved mask visualization to: {mask_path}")

if __name__ == "__main__":
    train()
