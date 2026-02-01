"""
Grad-CAM Implementation for Pneumonia Localization
- Captures gradients from the last conv layer (layer4 in ResNet18)
- Generates heatmap showing regions affected by pneumonia
- Red = infected lung areas, Blue = healthy lung areas
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM for Pneumonia Localization
    
    Always highlights regions that appear to have pneumonia signs,
    providing a clinically meaningful visualization:
    - Red/warm areas: Lung regions affected by pneumonia
    - Blue/cool areas: Healthy lung regions
    
    For ResNet18, we hook into 'layer4' (the last convolutional block)
    to capture feature maps and gradients.
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks to capture gradients and activations
        self._register_hooks()
    
    def _register_hooks(self):
        """Attach forward and backward hooks to target layer"""
        
        def forward_hook(module, input, output):
            # Store the output (feature maps) of the target layer
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            # Store the gradient flowing back through target layer
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM heatmap showing pneumonia-affected regions.
        
        Args:
            input_tensor: Preprocessed image tensor (1, 3, 224, 224)
            class_idx: Ignored - always uses pneumonia class for clinical relevance
        
        Returns:
            cam: Normalized heatmap (224, 224), values 0-1
                 High values (red) = pneumonia-affected regions
                 Low values (blue) = healthy regions
            prediction: Model's sigmoid output (pneumonia probability)
        """
        # Forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)
        prediction = torch.sigmoid(output).item()
        
        # Always compute gradients for pneumonia class (class 1)
        # This shows "where does the model see pneumonia signs" regardless of prediction
        # - Red areas: regions with pneumonia-like features
        # - Blue areas: healthy-looking regions
        output.backward()
        
        # Get gradients and activations
        gradients = self.gradients  # Shape: (1, 512, 7, 7)
        activations = self.activations  # Shape: (1, 512, 7, 7)
        
        # Global average pooling of gradients -> channel weights
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (1, 512, 1, 1)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1).squeeze()  # (7, 7)
        
        # ReLU - only keep positive contributions
        cam = F.relu(cam)
        
        # Normalize to 0-1 range
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Resize to input image size (224x224)
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        
        return cam, prediction


def create_heatmap_overlay(
    original_image: np.ndarray, 
    cam: np.ndarray, 
    lung_mask: np.ndarray = None,
    alpha: float = 0.6
):
    """
    Overlay Grad-CAM heatmap on original X-ray image.
    
    CRITICAL: Only shows colored overlay where BOTH conditions are met:
    1. CAM value is significant (above threshold)
    2. Pixel is inside the lung mask (if provided)
    
    Areas outside these conditions show the ORIGINAL image with NO color tint.
    
    Args:
        original_image: Original image as numpy array (H, W, 3), uint8
        cam: Grad-CAM heatmap (H, W), values 0-1
        lung_mask: Optional binary lung mask (H, W), values {0, 1}
        alpha: Blend factor for heatmap (0=original, 1=heatmap)
    
    Returns:
        overlay: Image with heatmap ONLY where CAM is significant and inside lungs
    """
    # Make a copy of original to avoid mutation
    img = original_image.copy()
    
    # Ensure original image is RGB uint8
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img = img.astype(np.uint8)
    
    h, w = img.shape[:2]
    
    # Resize CAM to match original image dimensions
    cam_resized = cv2.resize(cam.astype(np.float32), (w, h))
    cam_resized = np.clip(cam_resized, 0, 1)
    
    # Prepare lung mask (resize if provided)
    if lung_mask is not None:
        mask = cv2.resize(lung_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0.5).astype(np.float32)
    else:
        mask = np.ones((h, w), dtype=np.float32)
    
    # CRITICAL: Zero out CAM values OUTSIDE the lung mask
    cam_masked = cam_resized * mask
    
    # Create per-pixel alpha based on CAM intensity
    # This makes areas with low CAM completely transparent
    # HIGHER threshold = more selective overlay
    cam_threshold = 0.3  # Only show overlay where CAM > 30%
    pixel_alpha = np.clip((cam_masked - cam_threshold) / (1.0 - cam_threshold), 0, 1)
    pixel_alpha = pixel_alpha ** 0.7  # Power < 1 makes falloff smoother
    pixel_alpha = pixel_alpha * alpha  # Scale by global alpha
    
    # Convert CAM to hot colormap (better than JET - doesn't have blue for zeros)
    # Using custom colormap: transparent -> yellow -> orange -> red
    cam_uint8 = np.uint8(255 * cam_masked)
    
    # Use HOT colormap instead of JET (HOT: black -> red -> yellow -> white)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    
    # Apply per-pixel alpha blending
    # Where pixel_alpha = 0: show original image
    # Where pixel_alpha > 0: blend heatmap with original
    alpha_3ch = np.stack([pixel_alpha, pixel_alpha, pixel_alpha], axis=-1)
    
    img_float = img.astype(np.float32)
    heatmap_float = heatmap_rgb.astype(np.float32)
    
    overlay = img_float * (1.0 - alpha_3ch) + heatmap_float * alpha_3ch
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    return overlay

