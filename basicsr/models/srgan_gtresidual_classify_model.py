import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from thop import profile
from ptflops import get_model_complexity_info
import time

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img, savetensor, USMSharp
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel

from torch.autograd import gradcheck

@MODEL_REGISTRY.register()
class SRGANGTReidualClassifyModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SRGANGTReidualClassifyModel, self).__init__(opt)
        
        self.usm_sharpener = USMSharp().cuda()
        
        # Run DASR for making residual
        # self.net_g = build_network(opt['network_g'])
        # self.net_g = self.model_to_device(self.net_g)

        # self.net_p = build_network(opt['network_p'])
        # self.net_p = self.model_to_device(self.net_p)
        
        # # load pretrained models
        # load_path = self.opt['path'].get('pretrain_network_g', None)
        # load_key = self.opt['path'].get('param_key_g', None)
        # if load_path is not None:
        #     # if 'pretrained_models' in load_path and self.is_train:
        #     #     self.load_network_init_alldynamic(self.net_g, load_path, self.opt['num_networks'], self.opt['path'].get('strict_load_g', True), load_key)
        #     # else:
        #     self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), load_key)

        # load_path_p = self.opt['path'].get('pretrain_network_p', None)
        # if load_path_p is not None:
        #     self.load_network(self.net_p, load_path_p, self.opt['path'].get('strict_load_g', True))
            
        # self.net_p.eval()
        # self.net_g.eval()

        # define network
        self.net_c = build_network(opt['network_c'])
        self.net_c = self.model_to_device(self.net_c)
        
        self.road_map = [0,
                    10,
                    10 + 8,
                    10 + 8 + 8,
                    10 + 8 + 8 + 7]

        # load pretrained models
        load_path_c = self.opt['path'].get('pretrain_network_c', None)
        if load_path_c is not None:
            self.load_network(self.net_c, load_path_c, self.opt['path'].get('strict_load_g', True))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_c.train()
        train_opt = self.opt['train']

        # define losses

        if train_opt.get('regress_opt'):
            self.cri_regress = build_loss(train_opt['regress_opt']).to(self.device)
        else:
            self.cri_regress = None
            
        if train_opt.get('classify_opt'):
            self.cri_classify = build_loss(train_opt['classify_opt']).to(self.device)
        else:
            self.cri_classify = None

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_type = train_opt['optim_g'].pop('type')
        optim_params = []
        for k, v in self.net_c.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.degradation_params = data['degradation_params'].to(self.device)
        self.lq_path = data['lq_path']
        
        self.gt = data['gt'].to(self.device)
        self.gt = self.usm_sharpener(self.gt)
        
        self.degradation_classes = torch.zeros(self.lq.shape[0]).long()  # [B] 0~3, 0~2, 0~1 from 000 to 321
        if self.degradation_params[0, self.road_map[0]] < 0.5 and self.degradation_params[0, self.road_map[0]+1] < 0.5:
            self.degradation_classes[:] += 0*100
        elif self.degradation_params[0, self.road_map[0]] < 0.5 and self.degradation_params[0, self.road_map[0]+1] >= 0.5:
            self.degradation_classes[:] += 1*100
        elif self.degradation_params[0, self.road_map[0]] >= 0.5 and self.degradation_params[0, self.road_map[0]+1] < 0.5:
            self.degradation_classes[:] += 2*100
        else:
            self.degradation_classes[:] += 3*100
        if self.degradation_params[0, self.road_map[2]] < 0.33:
            self.degradation_classes[:] += 0*10
        elif self.degradation_params[0, self.road_map[2]] < 0.67:
            self.degradation_classes[:] += 1*10
        else:
            self.degradation_classes[:] += 2*10
        if self.degradation_params[0, self.road_map[3]] < 0.5:
            self.degradation_classes[:] += 0
        else:
            self.degradation_classes[:] += 1

    def optimize_parameters(self, current_iter):
        # Load DASR models
        # with torch.no_grad():
        #     _, weights = self.net_p(self.lq)
        #     sr_output = self.net_g(self.lq.contiguous(), weights)
        
        lr_h, lr_w = self.lq.shape[-2:]
        sr_output_ds = F.interpolate(self.gt, size=(lr_h, lr_w), mode='bicubic')
        lq_rd = self.lq - sr_output_ds
        
        # optimize net_c
        self.optimizer_g.zero_grad()
        out_params1, out_params2, out_params3 = self.net_c(lq_rd)
        out_label1, out_label2, out_label3 = self.degradation_classes // 100, (self.degradation_classes // 10) % 10, self.degradation_classes % 10

        l_g_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        # if self.cri_regress:
        #     l_regression = self.cri_regress(predicted_params, self.degradation_params)
        #     l_g_total += l_regression
        #     loss_dict['l_regression'] = l_regression
        if self.cri_classify:
            l_classification1 = self.cri_classify(out_params1, out_label1)
            #### Note: training only for blur classes
            l_classification2 = self.cri_classify(out_params2, out_label2)
            l_classification3 = self.cri_classify(out_params3, out_label3)
            l_g_total += l_classification1 + l_classification2 + l_classification3
            loss_dict['l_classification'] = l_classification1 + l_classification2 + l_classification3

        l_g_total.backward()
        self.optimizer_g.step()
        # breakpoint()
        self.log_dict = self.reduce_loss_dict(loss_dict)
        
        # For checking the number of data for each class
        out_label1 = F.one_hot(out_label1, num_classes=4).float().mean(dim=0).cpu()
        out_label2 = F.one_hot(out_label2, num_classes=3).float().mean(dim=0).cpu()
        out_label3 = F.one_hot(out_label3, num_classes=2).float().mean(dim=0).cpu()

        # print(out_label1, out_label2, out_label3)
        return out_label1, out_label2, out_label3

    def test(self):
        # Load DASR models
        # with torch.no_grad():
        #     _, weights = self.net_p(self.lq)
        #     sr_output = self.net_g(self.lq.contiguous(), weights)
        lr_h, lr_w = self.lq.shape[-2:]
        self.sr_output_ds = F.interpolate(self.gt, size=(lr_h, lr_w), mode='bicubic')
        self.lq_rd = self.lq - self.sr_output_ds
        
        self.net_c.eval()
        with torch.no_grad():
            pred_prob1, pred_prob2, pred_prob3 = self.net_c(self.lq_rd)
            # pred_prob1, pred_prob2, pred_prob3 = self.net_c(self.lq)
            
        self.pred_label1 = torch.argmax(pred_prob1, dim=-1)
        self.pred_label2 = torch.argmax(pred_prob2, dim=-1)
        self.pred_label3 = torch.argmax(pred_prob3, dim=-1)
        # self.predicted_params = predicted_params
        # self.weights = weights
        self.net_c.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        # with_metrics = self.opt['val'].get('metrics') is not None
        # if with_metrics:
        #     self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        pbar = tqdm(total=len(dataloader), unit='image')
        metric_results = [0, 0, 0]
        
        class_gt_blur, class_pred_blur = [0,0,0,0], [0,0,0,0]
        class_gt_noise, class_pred_noise = [0,0,0], [0,0,0]
        class_gt_jpeg, class_pred_jpeg = [0,0], [0,0]

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            # self.lq = val_data['lq'].to(self.device)
            # self.degradation_params = val_data['degradation_params'].to(self.device)
            # self.lq_path = val_data['lq_path']
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            lr_img = tensor2img([visuals['lq']])
            sr_img = tensor2img([visuals['lq_sr']])
            rd_img = tensor2img([visuals['lq_rd']])
            
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                h, w = sr_img.shape[:2]
                gt_img = gt_img[:h, :w]
                del self.gt

            if save_img:
                if self.opt['is_train']:
                    pass
                    save_img_path_rd = osp.join(self.opt['path']['visualization'], dataset_name, f'{current_iter}', 'residual', f'{img_name}.png')
                    save_img_path_sr = osp.join(self.opt['path']['visualization'], dataset_name, f'{current_iter}', 'lr',f'{img_name}.png')
                    imwrite(rd_img, save_img_path_rd)
                    imwrite(sr_img, save_img_path_sr)

                else:
                    if self.opt['val']['suffix']:
                        save_img_path_rd = osp.join(self.opt['path']['visualization']+'_residual', dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                        save_img_path_sr = osp.join(self.opt['path']['visualization']+'_lr', dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path_rd = osp.join(self.opt['path']['visualization']+'_residual', dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                        save_img_path_sr = osp.join(self.opt['path']['visualization']+'_lr', dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')

                    imwrite(rd_img, save_img_path_rd)
                    imwrite(lr_img, save_img_path_sr)
                # savetensor(self.predicted_params.cpu().squeeze(0), save_tensor_path)


            # if with_metrics:
            #     # calculate metrics
            #     for name, opt_ in self.opt['val']['metrics'].items():
            #         metric_data = dict(img1=sr_img, img2=gt_img)
            #         self.metric_results[name] += calculate_metric(metric_data, opt_)
            # metric_results += nn.L1Loss()(self.predicted_params, self.degradation_params)
            label1, label2, label3 = self.degradation_classes // 100, (self.degradation_classes // 10) % 10, self.degradation_classes % 10
            class_gt_blur[label1.item()] += 1
            class_gt_noise[label2.item()] += 1
            class_gt_jpeg[label3.item()] += 1
            
            class_pred_blur[self.pred_label1.item()] += 1
            class_pred_noise[self.pred_label2.item()] += 1
            class_pred_jpeg[self.pred_label3.item()] += 1
                        
            if label1.item() == self.pred_label1.item():
                metric_results[0] += 1
            if label2.item() == self.pred_label2.item():
                metric_results[1] += 1
            if label3.item() == self.pred_label3.item():
                metric_results[2] += 1

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
            # # tentative for out of GPU memory
            del self.lq
            del self.degradation_params
            del self.degradation_classes
            torch.cuda.empty_cache()
        pbar.close()
        
        metric_result = [ metric / (idx + 1) for metric in metric_results]
        logger = get_root_logger()
        logger.info(f'BlurKernel/{metric_result[0]}\tNoise/{metric_result[1]}\tJPEG/{metric_result[2]}')
        logger.info(f'BlurGT/{class_gt_blur}\tBlurPred/{class_pred_blur}\nNoiseGT/{class_gt_noise}\tNoisePred/{class_pred_noise}\nJpegGT/{class_gt_jpeg}\tJpegPred/{class_pred_jpeg}')

        # if with_metrics:
        #     for metric in self.metric_results.keys():
        #         self.metric_results[metric] /= (idx + 1)

        #     self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['lq_sr'] = self.sr_output_ds.detach().cpu()
        out_dict['lq_rd'] = self.lq_rd.detach().cpu()
        # if hasattr(self, 'gt'):
        #     out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_c, 'net_c', current_iter)
        self.save_training_state(epoch, current_iter)

