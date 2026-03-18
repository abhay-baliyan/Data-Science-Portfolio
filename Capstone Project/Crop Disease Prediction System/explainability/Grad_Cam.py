import torch
import numpy as np
import cv2
class GradCAM:
    def __init__(self,model,target_layer):
        self.model=model
        self.target_layer=target_layer
        self.gradients=None
        self.activations=None
        self._register_hooks()
    def _register_hooks(self):
        def forward_hook(module,input,output):
            self.activations=output.detach()
        def backward_hook(module,grad_in,grad_out):
            self.gradients=grad_out[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    def generate(self,input_tensor,class_idx):
        self.model.zero_grad()
        output=self.model(input_tensor)
        score=output[0,class_idx]
        score.backward()
        weights=self.gradients.mean(dim=(2,3),keepdim=True)
        cam=(weights*self.activations).sum(dim=1)
        cam=torch.relu(cam)
        cam=cam[0].cpu().numpy()
        cam=cv2.resize(cam,(224,224))
        cam=(cam-cam.min())/(cam.max()+1e-8)
        return cam