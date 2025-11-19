# lime_explainer.py
import numpy as np
import torch
from lime import lime_image
import cv2
from skimage.segmentation import mark_boundaries
from PIL import Image

class LimeWrapper:
    def __init__(self, model, preprocess_func, device, batch_predict_size=8):
        self.model = model
        self.preprocess = preprocess_func  # function to convert PIL -> tensor on device
        self.device = device
        self.batch = batch_predict_size

    def batch_predict(self, imgs):
        """
        imgs: list of numpy arrays (H,W,3) in RGB uint8
        returns: numpy array (N, num_classes) probabilities
        """
        from torchvision.transforms.functional import to_tensor
        xs = []
        for img in imgs:
            pil = Image.fromarray(img)
            tensor = self.preprocess(pil).unsqueeze(0)  # (1,3,H,W)
            xs.append(tensor)
        xs = torch.cat(xs, dim=0).to(self.device)
        with torch.no_grad():
            out = self.model(xs)
            probs = torch.softmax(out, dim=1).cpu().numpy()
        return probs

    def explain(self, pil_image, top_labels=(1,)):
        # convert pil to numpy rgb
        np_img = np.array(pil_image.convert('RGB'))
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            np_img,
            self.batch_predict,
            top_labels=top_labels,
            hide_color=0,
            num_samples=1000
        )
        # get superpixel mask for top label
        label = top_labels[0]
        temp, mask = explanation.get_image_and_mask(label, positive_only=True, num_features=10, hide_rest=False)
        # mark boundaries
        marked = mark_boundaries(temp / 255.0, mask)
        # convert to uint8
        marked = (marked * 255).astype(np.uint8)
        return Image.fromarray(marked)
