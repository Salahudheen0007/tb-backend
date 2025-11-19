# gradcam.py
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

def find_last_conv_layer(model):
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = (name, module)
    if last_conv is None:
        raise ValueError("No conv layer found")
    return last_conv[0]

class GradCAM:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
        # find last conv layer name
        self.target_layer_name = find_last_conv_layer(model)
        # hooks
        self.activations = None
        self.gradients = None
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        # register hooks
        for name, module in model.named_modules():
            if name == self.target_layer_name:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                break

    def __call__(self, input_tensor, target_class=None):
        # input_tensor: (1,3,H,W) on device
        output = self.model(input_tensor)            # logits
        probs = F.softmax(output, dim=1)
        if target_class is None:
            target_class = int(probs.argmax(dim=1).item())
        # backward for the target class
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward(retain_graph=True)

        # grads: (N, C, H, W) activations same shape
        grads = self.gradients      # (C, h, w)
        acts = self.activations     # (C, h, w)

        # global-average-pool grads -> weights
        weights = grads.mean(dim=(1,2))  # (C,)

        # weighted sum of activations
        cam = torch.zeros(acts.shape[1:], dtype=torch.float32, device=self.device)  # (h,w)
        for i, w in enumerate(weights):
            cam += w * acts[0, i]
        cam = cam.cpu().numpy()
        # relu
        cam = np.maximum(cam, 0)
        cam = cam - cam.min()
        if cam.max() != 0:
            cam = cam / cam.max()
        return cam, probs.detach().cpu().numpy()[0], target_class

def overlay_cam_on_image(pil_img, cam, colormap=cv2.COLORMAP_JET, alpha=0.4):
    # pil_img: PIL RGB
    img = np.array(pil_img.convert('RGB'))[:,:,::-1]  # BGR for cv2
    h, w = img.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = (cam_resized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    overlayed = cv2.addWeighted(heatmap, alpha, img, 1-alpha, 0)
    # convert back to RGB PIL
    overlayed = overlayed[:,:,::-1]
    return Image.fromarray(overlayed)
