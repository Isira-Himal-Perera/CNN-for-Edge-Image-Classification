"""Paired CNN designs: only the convolution operator differs."""
import torch
from torch import nn
from torchvision import models


class SeparableConv(nn.Module):
    """Depthwise 3x3 followed by pointwise 1x1; ReLU handled by caller."""
    def __init__(self, ins, outs):
        super().__init__()
        self.depthwise=nn.Conv2d(ins,ins,3,padding=1,groups=ins,bias=False)
        self.pointwise=nn.Conv2d(ins,outs,1,bias=True)
    def forward(self,x):return self.pointwise(self.depthwise(x))


class EdgeCNN(nn.Module):
    def __init__(self, num_classes=10, lightweight=False):
        super().__init__()
        conv = (lambda a,b: SeparableConv(a,b)) if lightweight else (lambda a,b: nn.Conv2d(a,b,3,padding=1))
        self.features=nn.Sequential(
            conv(3,16),nn.BatchNorm2d(16),nn.ReLU(inplace=True),nn.MaxPool2d(2),
            conv(16,32),nn.BatchNorm2d(32),nn.ReLU(inplace=True),nn.MaxPool2d(2),
            conv(32,64),nn.BatchNorm2d(64),nn.ReLU(inplace=True),nn.MaxPool2d(2),
        )
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.classifier=nn.Sequential(nn.Flatten(),nn.Linear(64,32),nn.ReLU(inplace=True),nn.Linear(32,num_classes))
    def forward(self,x):return self.classifier(self.pool(self.features(x)))


def load_pretrained(name,num_classes,allow_download=True):
    """Truly ImageNet PRETRAINED weights required: fail loudly rather than pretend."""
    if name == 'mobilenet_v2':
        w=models.MobileNet_V2_Weights.IMAGENET1K_V2
        m=models.mobilenet_v2(weights=w if allow_download else None)
        m.classifier[1]=nn.Linear(m.classifier[1].in_features,num_classes)
        backbone=m.features
        head=m.classifier
    elif name == 'squeezenet1_1':
        w=models.SqueezeNet1_1_Weights.IMAGENET1K_V1
        m=models.squeezenet1_1(weights=w if allow_download else None)
        m.classifier[1]=nn.Conv2d(512,num_classes,kernel_size=1)
        m.num_classes=num_classes
        backbone=m.features
        head=m.classifier
    else:raise ValueError(name)
    if not allow_download:
        raise RuntimeError('No pre-trained weights loaded. Cannot label an untrained network as fine-tuned.')
    m._pretrained_backbone=True
    for p in backbone.parameters():p.requires_grad=False
    for p in head.parameters():p.requires_grad=True
    return m


def unfreeze_last_block(model,which):
    """Fine-tune a small tail to limit compute and overfitting."""
    features=model.features
    n=3 if which=='mobilenet_v2' else 2
    for mod in list(features.children())[-n:]:
        for p in mod.parameters():p.requires_grad=True


def count_params(model,trainable_only=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def estimate_macs(model,image_size=32):
    """Convolution/linear multiply-accumulates for one forward pass, excluding activations/pooling."""
    macs=0
    hooks=[]
    def inspect(mod,inputs,output):
        nonlocal macs
        if isinstance(mod,nn.Conv2d):
            b,c,h,w=output.shape
            kh,kw=mod.kernel_size
            macs += c*h*w*(mod.in_channels//mod.groups)*kh*kw
        elif isinstance(mod,nn.Linear):
            macs += mod.in_features*mod.out_features
    for m in model.modules():
        if isinstance(m,(nn.Conv2d,nn.Linear)):hooks.append(m.register_forward_hook(inspect))
    model.eval()
    with torch.no_grad():model(torch.zeros(1,3,image_size,image_size))
    for hook in hooks:hook.remove()
    return int(macs)
