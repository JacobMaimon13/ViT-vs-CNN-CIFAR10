import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def save_gradcam(model, image_tensor, label, pred, device, save_path="outputs/figures/grad_cam.png"):
    """
    Generates and saves a Grad-CAM heatmap for the CNN model.
    """
    model.eval()
    model.to(device)
    
    # Target Layer: Usually the last convolutional layer (conv3 in our SimpleCNN)
    target_layers = [model.conv3]
    
    # Prepare image
    input_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Initialize GradCAM
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Generate CAM
    targets = [ClassifierOutputTarget(pred)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    
    # Denormalize image for visualization
    rgb_img = image_tensor.permute(1, 2, 0).cpu().numpy()
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
    
    # Overlay
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    # Plot & Save
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_img)
    plt.title(f"Original (True: {label})")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title(f"Grad-CAM (Pred: {pred})")
    plt.axis('off')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved Grad-CAM to {save_path}")

def save_attention_map(model, image_tensor, label, pred, device, save_path="outputs/figures/attention_map.png"):
    """
    Generates and saves Attention Map for the ViT model.
    """
    model.eval()
    model.to(device)
    
    # Prepare input
    x = image_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        # 1. Get Patch Embeddings
        patches = model.model.patch_embed(x)
        B, N, E = patches.shape
        
        # 2. Add CLS token & Positional Embedding
        cls_token = model.model.cls_token.expand(B, -1, -1)
        x_emb = torch.cat((cls_token, patches), dim=1)
        x_emb = x_emb + model.model.pos_embed
        
        # 3. Extract Attention from the last Transformer Encoder Layer
        # Note: We need to hook into the MultiheadAttention to get weights
        last_layer = model.model.encoder.layers[-1]
        
        # Forward pass through self-attention to capture weights
        # (This relies on standard PyTorch Transformer implementation details)
        attn_output, attn_weights = last_layer.self_attn(x_emb, x_emb, x_emb, average_attn_weights=True)
        
        # Take attention of CLS token (index 0) to all other tokens (1 to N)
        cls_attention = attn_weights[0, 0, 1:] # Shape: [N_patches]
        
        # Reshape to grid
        grid_size = int(np.sqrt(N))
        attn_grid = cls_attention.reshape(grid_size, grid_size).cpu().numpy()
        
        # Resize to original image size
        img_h, img_w = image_tensor.shape[1], image_tensor.shape[2]
        attn_resized = cv2.resize(attn_grid, (img_w, img_h))
        attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min())

    # Visualization
    rgb_img = image_tensor.permute(1, 2, 0).cpu().numpy()
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_img)
    plt.title(f"Original (True: {label})")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(rgb_img)
    plt.imshow(attn_resized, cmap='jet', alpha=0.5) # Overlay heatmap
    plt.title(f"ViT Attention Map")
    plt.axis('off')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved Attention Map to {save_path}")
