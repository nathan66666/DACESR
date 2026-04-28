import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from thop import profile
from ptflops import get_model_complexity_info
import time
import math
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
from ram.models.ram_lora import ram as ram
from torchvision import transforms
from torch.autograd import gradcheck

@MODEL_REGISTRY.register()
class DACESR_test_Model(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(DACESR_test_Model, self).__init__(opt)

        # define network
        if opt['ram_model_path1'] == 'None':
            opt['ram_model_path1'] = None
        if opt['ram_model_path2'] == 'None':
            opt['ram_model_path2'] = None

        if opt['ram_model_path3'] == 'None':
            opt['ram_model_path3'] = None

        if opt['ram_model_path4'] == 'None':
            opt['ram_model_path4'] = None

        if opt['ram_model_path1'] is not None:
            self.net_r1 = ram(pretrained=opt['ram_model_path'], pretrained_condition=opt['ram_model_path1'], image_size=384, vit='swin_l')
            self.net_r1 = self.model_to_device(self.net_r1)
            self.net_r1.eval()
        if opt['ram_model_path2'] is not None:
            self.net_r2 = ram(pretrained=opt['ram_model_path'], pretrained_condition=opt['ram_model_path2'], image_size=384, vit='swin_l')
            self.net_r2 = self.model_to_device(self.net_r2)
            self.net_r2.eval()
        if opt['ram_model_path3'] is not None:
            self.net_r3 = ram(pretrained=opt['ram_model_path'], pretrained_condition=opt['ram_model_path3'], image_size=384, vit='swin_l')
            self.net_r3 = self.model_to_device(self.net_r3)
            self.net_r3.eval()
        if opt['ram_model_path4'] is not None:
            self.net_r4 = ram(pretrained=opt['ram_model_path'], pretrained_condition=opt['ram_model_path4'], image_size=384, vit='swin_l')
            self.net_r4 = self.model_to_device(self.net_r4)
            self.net_r4.eval()


        # Mamba
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)


        if self.is_train:
            self.net_d = build_network(self.opt['network_d'])
            self.net_d = self.model_to_device(self.net_d)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        load_key = self.opt['path'].get('param_key_g', None)
        if load_path is not None:
            if 'pretrained_models' in load_path and self.is_train:
                self.load_network_init_alldynamic(self.net_g, load_path, self.opt['num_networks'], self.opt['path'].get('strict_load_g', True), load_key)
            else:
                self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), load_key)
                # self.load_network_init_alldynamic(self.net_g, load_path, self.opt['num_networks'], self.opt['path'].get('strict_load_g', True), load_key)

        if self.is_train:
            load_path_d = self.opt['path'].get('pretrain_network_d', None)
            load_key = self.opt['path'].get('param_key_g', None)
            if load_path_d is not None:
                self.load_network(self.net_d, load_path_d, self.opt['path'].get('strict_load_d', True), load_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        self.net_d.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            for p in self.net_g_ema.parameters():
                p.requires_grad = False
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

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

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_type = train_opt['optim_g'].pop('type')
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()

        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        optim_type = train_opt['optim_d'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        lr_image = transforms.functional.resize(self.lq, (512, 512))
        lr_image_ram = transforms.functional.resize(lr_image, (384, 384))
        self.lq_ram = transforms.functional.normalize(lr_image_ram, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(self.device)
                
        # self.lq_ram = data['lq_ram'].to(self.device)
        # self.gt_ram = data['gt_ram'].to(self.device)
        self.lq_path = data['lq_path']
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):

        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()

        if hasattr(self, 'net_r1') and self.net_r1 is not None:
            feature_lq1, logits_lq1, _ = self.net_r1.condition_forward(self.lq_ram, only_feature=False)
            self.output_c = feature_lq1

        if hasattr(self, 'net_r2') and self.net_r2 is not None:
            feature_lq2, logits_lq2, _ = self.net_r2.condition_forward(self.lq_ram, only_feature=False)
            self.output_c += feature_lq2

        if hasattr(self, 'net_r3') and self.net_r3 is not None:
            feature_lq3, logits_lq3, _ = self.net_r3.condition_forward(self.lq_ram, only_feature=False)
            self.output_c += feature_lq3

        if hasattr(self, 'net_r4') and self.net_r4 is not None:
            feature_lq4, logits_lq4, _ = self.net_r4.condition_forward(self.lq_ram, only_feature=False)
            self.output_c += feature_lq4 
        self.output = self.net_g(self.lq, self.output_c)


        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_pix = self.cri_pix(self.output, self.gt)
                l_g_total += l_pix
                loss_dict['l_pix'] = l_pix
            # perceptual loss
            if self.cri_perceptual:
                l_percep, l_style = self.cri_perceptual(self.output, self.gt)
                if l_percep is not None:
                    l_g_total += l_percep
                    loss_dict['l_percep'] = l_percep
                if l_style is not None:
                    l_g_total += l_style
                    loss_dict['l_style'] = l_style
            # gan loss
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # real
        real_d_pred = self.net_d(self.gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward()
        # fake
        fake_d_pred = self.net_d(self.output.detach())
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()
        self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
    def process_net_r(self):
            output_c = 0

            if hasattr(self, 'net_r1') and self.net_r1 is not None:
                feature_lq1, logits_lq1, _ = self.net_r1.condition_forward(self.lq_ram, only_feature=False)
                output_c += feature_lq1 

            if hasattr(self, 'net_r2') and self.net_r2 is not None:
                feature_lq2, logits_lq2, _ = self.net_r2.condition_forward(self.lq_ram, only_feature=False)
                output_c += feature_lq2 
            if hasattr(self, 'net_r3') and self.net_r3 is not None:
                feature_lq3, logits_lq3, _ = self.net_r3.condition_forward(self.lq_ram, only_feature=False)
                output_c += feature_lq3 

            if hasattr(self, 'net_r4') and self.net_r4 is not None:
                feature_lq4, logits_lq4, _ = self.net_r4.condition_forward(self.lq_ram, only_feature=False)
                output_c += feature_lq4 

            return output_c

    def test(self):

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                output_c = self.process_net_r()
                self.output = self.net_g_ema(self.lq, output_c)
        else:
            self.net_g.eval()
            with torch.no_grad():
                output_c = self.process_net_r()
                self.output = self.net_g(self.lq, output_c)
            self.net_g.train()


    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)
    def pre_process(self):
        self.scale = self.opt.get('scale', 1)
        self.img = self.lq
        

    def process(self):
        # model inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.img)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.img)
            # self.net_g.train()

    def tile_process(self):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.opt['tile']['tile_size'])
        tiles_y = math.ceil(height / self.opt['tile']['tile_size'])

        # Loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # Extract tile from input image
                ofs_x = x * self.opt['tile']['tile_size']
                ofs_y = y * self.opt['tile']['tile_size']
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.opt['tile']['tile_size'], width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.opt['tile']['tile_size'], height)

                input_start_x_pad = max(input_start_x - self.opt['tile']['tile_pad'], 0)
                input_end_x_pad = min(input_end_x + self.opt['tile']['tile_pad'], width)
                input_start_y_pad = max(input_start_y - self.opt['tile']['tile_pad'], 0)
                input_end_y_pad = min(input_end_y + self.opt['tile']['tile_pad'], height)

                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # Try to upscale tile
                try:
                    if hasattr(self, 'net_g_ema'):
                        self.net_g_ema.eval()
                        with torch.no_grad():
                            output_c = self.process_net_r()
                            output_tile = self.net_g_ema(input_tile, output_c)
                    else:
                        self.net_g.eval()
                        with torch.no_grad():
                            output_c = self.process_net_r()
                            output_tile = self.net_g(input_tile, output_c)
                except RuntimeError as error:
                    print(f'Error in processing tile {tile_idx}:', error)
                    output_tile = None  # Set output_tile to None to skip processing this tile
                    continue  # Skip to the next tile if there was an error

                print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # Calculate output tile area on total image
                output_start_x = input_start_x * self.opt['scale']
                output_end_x = input_end_x * self.opt['scale']
                output_start_y = input_start_y * self.opt['scale']
                output_end_y = input_end_y * self.opt['scale']

                output_start_x_tile = (input_start_x - input_start_x_pad) * self.opt['scale']
                output_end_x_tile = output_start_x_tile + input_tile_width * self.opt['scale']
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.opt['scale']
                output_end_y_tile = output_start_y_tile + input_tile_height * self.opt['scale']

                # Ensure output_tile is not None before placing it in the output image
                if output_tile is not None:
                    self.output[:, :, output_start_y:output_end_y,
                                output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                        output_start_x_tile:output_end_x_tile]
                else:
                    print(f'Skipping tile {tile_idx} due to an error in processing.')

    def _initialize_best_metric_results(self, dataset_name):
        """Initialize the best metric results dict for recording the best metric value and iteration."""
        if hasattr(self, 'best_metric_results') and dataset_name in self.best_metric_results:
            return
        elif not hasattr(self, 'best_metric_results'):
            self.best_metric_results = dict()

        # add a dataset record
        record = dict()
        for metric, content in self.opt['val']['metrics'].items():
            better = content.get('better', 'higher')
            init_val = float('-inf') if better == 'higher' else float('inf')
            record[metric] = dict(better=better, val=init_val, iter=-1)
        self.best_metric_results[dataset_name] = record

    def _update_best_metric_result(self, dataset_name, metric, val, current_iter):
        if self.best_metric_results[dataset_name][metric]['better'] == 'higher':
            if val >= self.best_metric_results[dataset_name][metric]['val']:
                self.best_metric_results[dataset_name][metric]['val'] = val
                self.best_metric_results[dataset_name][metric]['iter'] = current_iter
        else:
            if val <= self.best_metric_results[dataset_name][metric]['val']:
                self.best_metric_results[dataset_name][metric]['val'] = val
                self.best_metric_results[dataset_name][metric]['iter'] = current_iter
    def post_process(self):
        # self.output = self.output
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)

            self.pre_process()
            if 'tile' in self.opt:
                self.tile_process()
            else:
                self.process()
            # self.post_process()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img1'] = sr_img
            # print(sr_img.shape)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                # print(gt_img.shape)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

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
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)

