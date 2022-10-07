from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.ffc import FFCBlock
import torch.nn as nn

@ARCH_REGISTRY.register()
class FFC_Degradation_Classifier(nn.Module):
    def __init__(self, in_nc=3, nf=64, use_bias=True):
        super(FFC_Degradation_Classifier, self).__init__()

        self.ConvNet = nn.Sequential(*[
            nn.Conv2d(in_nc, nf, kernel_size=5, stride=1, padding=2),
            FFCBlock(inplanes=nf, planes=nf, ratio_gin=0,),
            FFCBlock(inplanes=nf, planes=nf),
            FFCBlock(inplanes=nf, planes=nf),
            FFCBlock(inplanes=nf, planes=nf),
        ])
        
        self.globalPooling = nn.AdaptiveAvgPool2d((1, 1))
        
        self.ConvBR1 = nn.Sequential(*[
            FFCBlock(inplanes=nf, planes=nf),
            FFCBlock(inplanes=nf, planes=nf, ratio_gout=0),
        ])
        self.Mapping1 = nn.Sequential(*[
            nn.Linear(nf, 15),
            nn.Linear(15, 4),
        ])

        self.ConvBR2 = nn.Sequential(*[
            FFCBlock(inplanes=nf, planes=nf),
            FFCBlock(inplanes=nf, planes=nf, ratio_gout=0),
        ])
        self.Mapping2 = nn.Sequential(*[
            nn.Linear(nf, 15),
            nn.Linear(15, 3),
        ])

        self.ConvBR3 = nn.Sequential(*[
            FFCBlock(inplanes=nf, planes=nf),
            FFCBlock(inplanes=nf, planes=nf, ratio_gout=0),
        ])
        self.Mapping3 = nn.Sequential(*[
            nn.Linear(nf, 15),
            nn.Linear(15, 2),
        ])


    def forward(self, input):
        conv = self.ConvNet(input)
        
        conv1 = self.ConvBR1(conv)
        flat1 = self.globalPooling(conv1[0])
        flat1 = flat1.view(flat1.size()[:2])
        out_params1 = self.Mapping1(flat1)
        
        conv2 = self.ConvBR2(conv)
        flat2 = self.globalPooling(conv2[0])
        flat2 = flat2.view(flat2.size()[:2])
        out_params2 = self.Mapping2(flat2)
        
        conv3 = self.ConvBR3(conv)
        flat3 = self.globalPooling(conv3[0])
        flat3 = flat3.view(flat3.size()[:2])
        out_params3 = self.Mapping3(flat3)
        
        return out_params1, out_params2, out_params3
