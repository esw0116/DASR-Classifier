from basicsr.utils.registry import ARCH_REGISTRY
import torch
import torch.nn as nn


@ARCH_REGISTRY.register()
class Degradation_Classifier2(nn.Module):
    def __init__(self, in_nc=3, nf=64, use_bias=True):
        super(Degradation_Classifier2, self).__init__()

        self.norm1 = LayerNorm2d(in_nc)
        # self.norm1 = nn.BatchNorm2d(in_nc)

        self.ConvNet = nn.Sequential(*[
            nn.Conv2d(in_nc, nf, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.05, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.05, True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.05, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=2, padding=2, bias=use_bias),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.05, True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        
        # self.norm2 = LayerNorm2d(nf)
        # self.norm2 = nn.BatchNorm2d(nf)

        self.globalPooling = nn.AdaptiveAvgPool2d((6, 6))
        
        self.ConvBR1 = nn.Sequential(*[
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.05, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.05, True),
        ])
        # self.normbr1 = nn.LayerNorm(nf)

        self.Mapping1 = nn.Sequential(*[
            nn.Linear(nf*36, 64),
            # nn.ReLU(True),
            nn.Linear(64, 4),
        ])

        self.ConvBR2 = nn.Sequential(*[
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.05, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.05, True),
        ])
        self.normbr2 = nn.LayerNorm(nf)
        self.Mapping2 = nn.Sequential(*[
            nn.Linear(nf*36, 64),
            nn.ReLU(True),
            nn.Linear(64, 3),
        ])

        self.ConvBR3 = nn.Sequential(*[
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.05, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.05, True),
        ])
        # self.normbr3 = nn.LayerNorm(nf)
        self.Mapping3 = nn.Sequential(*[
            nn.Linear(nf*36, 64),
            # nn.ReLU(True),
            nn.Linear(64, 2),
        ])
        
        if True:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)


    def forward(self, img):
        # img = img.permute(0,2,3,1)
        norm_img = self.norm1(img)
        # norm_img = norm_img.permute(0,3,1,2)
        # norm_img = norm_img * 5
        # norm_img = img
        conv = self.ConvNet(norm_img)
        # conv = conv.permute(0,2,3,1)
        # norm_conv = self.norm2(conv)
        # norm_conv = norm_conv.permute(0,3,1,2)
        norm_conv = conv
        
        conv1 = self.ConvBR1(norm_conv)
        # conv1 = conv1.permute(0,2,3,1)
        # norm_conv1 = self.normbr1(conv1)
        # # norm_conv1 = norm_conv1 * 1
        # norm_conv1 = norm_conv1.permute(0,3,1,2)
        norm_conv1 = conv1
        flat1 = self.globalPooling(norm_conv1)
        flat1 = torch.flatten(flat1, 1)
        out_params1 = self.Mapping1(flat1)
        
        conv2 = self.ConvBR2(norm_conv)
        conv2 = conv2.permute(0,2,3,1)
        norm_conv2 = self.normbr2(conv2)
        # norm_conv2 = norm_conv2 * 4
        norm_conv2 = norm_conv2.permute(0,3,1,2)
        # norm_conv2 = conv2
        flat2 = self.globalPooling(norm_conv2)
        flat2 = torch.flatten(flat2, 1)
        out_params2 = self.Mapping2(flat2)
        
        conv3 = self.ConvBR3(norm_conv)
        # conv3 = conv3.permute(0,2,3,1)
        # norm_conv3 = self.normbr3(conv3)
        # # norm_conv3 = norm_conv3 * 10
        # norm_conv3 = norm_conv3.permute(0,3,1,2)
        norm_conv3 = conv3
        flat3 = self.globalPooling(norm_conv3)
        flat3 = torch.flatten(flat3, 1)
        out_params3 = self.Mapping3(flat3)
        
        # breakpoint()
        
        return out_params1, out_params2, out_params3


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)