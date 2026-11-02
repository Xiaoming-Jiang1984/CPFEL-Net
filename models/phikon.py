import torch
import torch.nn as nn
import timm
import sys
sys.path.append('/mnt/SSD_4T/admin103/ydw/UNI_classify/models')
from Phikon_lib.HistoSSLscaling.rl_benchmarks.models import iBOTViT

class phikon_freeze(nn.Module):
    def __init__(self, checkpoint='/mnt/SSD_4T/admin103/ydw/UNI_classify/models/Phikon_lib/ibot_vit_base_pancan.pth', num_classes=2):
        super(phikon_freeze, self).__init__()

        self.basemodel = iBOTViT(architecture="vit_base_pancan", encoder="teacher", weights_path=checkpoint)

        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
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
    model = phikon_freeze(num_classes=12)
    model.print_freeze()