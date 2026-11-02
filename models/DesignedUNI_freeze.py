# from .trans_attention import Attention

import torch
from torchvision import transforms
import torch.nn as nn

import timm

class designedUNI_freeze(nn.Module):
    def __init__(self, checkpoint_pth='./pretrained_weights/UNI/pytorch_model.bin', num_classes=2):
        super(designedUNI_freeze, self).__init__()

        self.basemodel = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )

        print('load UNI checkpoints')
        checkpoint = torch.load(checkpoint_pth, map_location='cpu')
        weights_dict = {}
        weights = checkpoint
        for k, v in weights.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            weights_dict[new_k] = v
        msg = self.basemodel.load_state_dict(weights_dict, strict=False)
        print('UNI load msg:', msg)

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=-1),
        )

        self.freeze_UNI_base()
        self.print_freeze()

    def freeze_UNI_base(self):
        print('freeze_UNI_base')
        for nm, pm in self.basemodel.named_parameters():
            pm.requires_grad = False
        self.basemodel.eval()

    def print_freeze(self):
        print('learnable parameters')
        for nm, pm in self.named_parameters():
            if pm.requires_grad:
                print(nm, pm.requires_grad)
        print('Done', flush=True)

    def forward(self, x):
        h = self.basemodel(x)
        # cls = self.classifier(h)
        return h


if __name__ == '__main__':
    model = designedUNI_freeze(num_classes=12)
    model.print_freeze()
