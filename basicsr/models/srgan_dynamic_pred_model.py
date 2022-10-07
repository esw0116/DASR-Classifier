import torch
from torch import nn
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from thop import profile
from ptflops import get_model_complexity_info
import time

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img, savetensor
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel

from torch.autograd import gradcheck

@MODEL_REGISTRY.register()
class SRGANPredModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SRGANPredModel, self).__init__(opt)

        # define network
        self.net_p = build_network(opt['network_p'])
        self.net_p = self.model_to_device(self.net_p)

        # load pretrained models
        load_path_p = self.opt['path'].get('pretrain_network_p', None)
        if load_path_p is not None:
            self.load_network(self.net_p, load_path_p, self.opt['path'].get('strict_load_g', True))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_p.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('regress_opt'):
            self.cri_regress = build_loss(train_opt['regress_opt']).to(self.device)
        else:
            self.cri_regress = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_type = train_opt['optim_g'].pop('type')
        optim_params = []
        for k, v in self.net_p.named_parameters():
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

    def optimize_parameters(self, current_iter):
        # optimize net_p
        self.optimizer_g.zero_grad()
        predicted_params, weights = self.net_p(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_regress:
            l_regression = self.cri_regress(predicted_params, self.degradation_params)
            l_g_total += l_regression
            loss_dict['l_regression'] = l_regression

        l_g_total.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_p.eval()
        with torch.no_grad():
            predicted_params, weights = self.net_p(self.lq)
        self.predicted_params = predicted_params
        self.weights = weights
        self.net_p.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        # with_metrics = self.opt['val'].get('metrics') is not None
        # if with_metrics:
        #     self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        pbar = tqdm(total=len(dataloader), unit='image')
        metric_results = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.lq = val_data['lq'].to(self.device)
            self.degradation_params = val_data['degradation_params'].to(self.device)
            self.lq_path = val_data['lq_path']
            self.test()

            # visuals = self.get_current_visuals()
            # sr_img = tensor2img([visuals['result']])
            # if 'gt' in visuals:
            #     gt_img = tensor2img([visuals['gt']])
            #     h, w = sr_img.shape[:2]
            #     gt_img = gt_img[:h, :w]
            #     del self.gt

            # if save_img:
            #     if self.opt['is_train']:
            #         save_img_path = osp.join(self.opt['path']['visualization'], img_name, f'{img_name}_{current_iter}.png')
            #     else:
            #         if self.opt['val']['suffix']:
            #             save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
            #                                      f'{img_name}_{self.opt["val"]["suffix"]}.png')
            #             save_tensor_path = osp.join(self.opt['path']['visualization']+'_degradation', dataset_name,
            #                                      f'{img_name}_{self.opt["val"]["suffix"]}.pt')
            #         else:
            #             save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
            #                                      f'{img_name}_{self.opt["name"]}.png')
            #             save_tensor_path = osp.join(self.opt['path']['visualization']+'_degradation', dataset_name,
            #                                      f'{img_name}_{self.opt["name"]}.pt')

            #     imwrite(sr_img, save_img_path)
            #     savetensor(self.predicted_params.cpu().squeeze(0), save_tensor_path)


            # if with_metrics:
            #     # calculate metrics
            #     for name, opt_ in self.opt['val']['metrics'].items():
            #         metric_data = dict(img1=sr_img, img2=gt_img)
            #         self.metric_results[name] += calculate_metric(metric_data, opt_)
            metric_results += nn.L1Loss()(self.predicted_params, self.degradation_params)
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
            # # tentative for out of GPU memory
            del self.lq
            del self.degradation_params
            torch.cuda.empty_cache()
        pbar.close()
        
        metric_results /= (idx + 1)
        logger = get_root_logger()
        logger.info(f'L1dist/{metric_results}')

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
        # out_dict['result'] = self.output.detach().cpu()
        # if hasattr(self, 'gt'):
        #     out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_p, 'net_p', current_iter)
        self.save_training_state(epoch, current_iter)

