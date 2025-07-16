import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.srgan_model import SRGANModel  # ← Changed this line
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils import tensor2img, imwrite      # ← Added this line
from collections import OrderedDict
import numpy as np
import os.path as osp                              # ← Added this line
from tqdm import tqdm                              # ← Added this line
from basicsr.losses.basic_loss import l1_loss, mse_loss

# Импортируем модифицированные версии
from models.network_swinir import SwinIR
from realesrgan.archs.discriminator_arch import UNetDiscriminatorSN
from utils import calculate_psnr, calculate_ssim

class TemperaturePerceptualLoss(nn.Module):
    """Perceptual loss адаптированный для температурных данных"""

    def __init__(self, feature_weights=None):
        super().__init__()
        if feature_weights is None:
            self.feature_weights = [1.0, 1.0, 1.0, 1.0]
        else:
            self.feature_weights = feature_weights

        # Создаем простую сеть для извлечения признаков из температурных данных
        self.feature_extractor = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 32, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, 2, 1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, 2, 1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, 2, 1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, 2, 1),
                nn.ReLU(inplace=True)
            )
        ])

        # Заморозим веса для стабильности
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        """Вычисление perceptual loss между x и y"""
        loss = 0

        # Проходим через каждый уровень feature extractor
        feat_x = x
        feat_y = y

        for i, layer in enumerate(self.feature_extractor):
            feat_x = layer(feat_x)
            feat_y = layer(feat_y)

            # L1 loss между признаками
            loss += self.feature_weights[i] * F.l1_loss(feat_x, feat_y)

        return loss


class PhysicsConsistencyLoss(nn.Module):
    """Loss для сохранения физической консистентности температурных данных"""

    def __init__(self, gradient_weight=0.1, smoothness_weight=0.05):
        super().__init__()
        self.gradient_weight = gradient_weight
        self.smoothness_weight = smoothness_weight

    def forward(self, pred, target):
        """
        Вычисляет loss с учетом физических свойств температурного поля
        """
        # Основной L1 loss
        main_loss = F.l1_loss(pred, target)

        # Градиентный loss - сохраняем резкость границ
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        gradient_loss = F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)

        # Smoothness loss - избегаем артефактов
        smooth_x = pred[:, :, :, 1:] - 2 * pred[:, :, :, :-1] + pred[:, :, :, :-1]
        smooth_y = pred[:, :, 1:, :] - 2 * pred[:, :, :-1, :] + pred[:, :, :-1, :]
        smoothness_loss = torch.mean(torch.abs(smooth_x)) + torch.mean(torch.abs(smooth_y))

        total_loss = main_loss + self.gradient_weight * gradient_loss + self.smoothness_weight * smoothness_loss

        return total_loss, {
            'main': main_loss,
            'gradient': gradient_loss,
            'smoothness': smoothness_loss
        }


@MODEL_REGISTRY.register()
class TemperatureSRModel(SRGANModel):
    """Гибридная модель для Super-Resolution температурных данных"""

    def __init__(self, opt):
        # Initialize base attributes that parent class expects
        self.opt = opt
        self.device = torch.device('cuda' if opt.get('num_gpu', 0) > 0 else 'cpu')
        self.is_train = opt.get('is_train', True)

        # Initialize required lists
        self.optimizers = []
        self.schedulers = []

        # Build our custom generator
        self.net_g = self.build_swinir_generator(opt)
        self.net_g = self.net_g.to(self.device)
        self.print_network = lambda x: print(
            f"Network: {x.__class__.__name__}, parameters: {sum(p.numel() for p in x.parameters()):,}")
        self.print_network(self.net_g)

        # Build discriminator BEFORE training setup
        if self.is_train:
            self.net_d = UNetDiscriminatorSN(
                num_in_ch=1,
                num_feat=opt['network_d'].get('num_feat', 64),
                skip_connection=opt['network_d'].get('skip_connection', True)
            )
            self.net_d = self.net_d.to(self.device)
            self.print_network(self.net_d)

            # Now initialize training settings
            self.init_training_settings()

    def build_swinir_generator(self, opt):
        """Построение SwinIR генератора для температурных данных"""
        opt_net = opt['network_g']

        # Параметры для 8x увеличения с 1 каналом
        model = SwinIR(
            upscale=2,  # 8x увеличение
            in_chans=1,  # 1 канал входа
            img_size=opt_net.get('img_size', 64),
            window_size=opt_net.get('window_size', 8),
            img_range=1.,
            depths=opt_net.get('depths', [8, 8, 8, 8, 8, 8, 8, 8]),
            embed_dim=opt_net.get('embed_dim', 240),
            num_heads=opt_net.get('num_heads', [8, 8, 8, 8, 8, 8, 8, 8]),
            mlp_ratio=opt_net.get('mlp_ratio', 4),
            upsampler=opt_net.get('upsampler', 'pixelshuffle'),
            resi_connection=opt_net.get('resi_connection', '3conv')
        )

        return model

    def init_training_settings(self):
        """Инициализация настроек обучения с физическими losses"""
        train_opt = self.opt['train']

        # Настройка pixel loss
        if train_opt.get('pixel_opt'):
            self.cri_pix = PhysicsConsistencyLoss(
                gradient_weight=train_opt['pixel_opt'].get('gradient_weight', 0.1),
                smoothness_weight=train_opt['pixel_opt'].get('smoothness_weight', 0.05)
            )
        else:
            self.cri_pix = None

        # Настройка perceptual loss для температур
        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = TemperaturePerceptualLoss(
                feature_weights=train_opt['perceptual_opt'].get('feature_weights', [1.0, 1.0, 1.0, 1.0])
            ).to(self.device)
        else:
            self.cri_perceptual = None

        # GAN loss
        if train_opt.get('gan_opt'):
            from basicsr.losses.gan_loss import GANLoss
            self.cri_gan = GANLoss(
                train_opt['gan_opt']['gan_type'],
                real_label_val=1.0,
                fake_label_val=0.0,
                loss_weight=train_opt['gan_opt']['loss_weight']
            ).to(self.device)
        else:
            self.cri_gan = None

        # Set training parameters
        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # Настройка оптимизаторов
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_schedulers(self):
        """Set up schedulers."""
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')

        if scheduler_type == 'MultiStepLR':
            from torch.optim.lr_scheduler import MultiStepLR
            for optimizer in self.optimizers:
                self.schedulers.append(
                    MultiStepLR(optimizer, **train_opt['scheduler'])
                )

    def optimize_parameters(self, current_iter):
        """Оптимизация с учетом физических ограничений"""

        # Clear cache at start
        torch.cuda.empty_cache()

        # ============ ГЕНЕРАТОР ОБУЧАЕТСЯ ВСЕГДА ============
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()

        # Pixel loss - ВСЕГДА
        if self.cri_pix:
            l_g_pix, pix_losses = self.cri_pix(self.output, self.gt)
            l_g_total += l_g_pix * self.opt['train']['pixel_opt']['loss_weight']
            loss_dict['l_g_pix'] = l_g_pix
            for k, v in pix_losses.items():
                loss_dict[f'l_g_pix_{k}'] = v

        # Perceptual loss - ВСЕГДА
        if self.cri_perceptual:
            l_g_percep = self.cri_perceptual(self.output, self.gt)
            l_g_total += l_g_percep * self.opt['train']['perceptual_opt']['loss_weight']
            loss_dict['l_g_percep'] = l_g_percep

        # GAN loss - только после прогрева
        if self.cri_gan and current_iter > self.net_d_init_iters:
            # Detach and clone to reduce memory
            output_for_d = self.output.detach().clone()
            fake_g_pred = self.net_d(output_for_d)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan
            # Clear immediately
            del fake_g_pred, output_for_d

        # ВСЕГДА обновляем генератор
        l_g_total.backward()

        # Gradient clipping
        if self.opt['train'].get('use_grad_clip', True):
            torch.nn.utils.clip_grad_norm_(
                self.net_g.parameters(),
                max_norm=self.opt['train'].get('grad_clip_norm', 20.0)  # Увеличено с 0.1
            )


        self.optimizer_g.step()

        del l_g_total
        torch.cuda.empty_cache()

        # ============ ДИСКРИМИНАТОР ============
        if current_iter > self.net_d_init_iters and current_iter % self.net_d_iters == 0:
            for p in self.net_d.parameters():
                p.requires_grad = True

            self.optimizer_d.zero_grad()

            # Real
            real_d_pred = self.net_d(self.gt.detach())  # добавил .detach()
            l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
            loss_dict['l_d_real'] = l_d_real
            loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
            l_d_real.backward()
            del real_d_pred, l_d_real

            # Fake
            fake_d_pred = self.net_d(self.output.detach())
            l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
            loss_dict['l_d_fake'] = l_d_fake
            loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
            l_d_fake.backward()
            del fake_d_pred, l_d_fake

            self.optimizer_d.step()

            torch.cuda.empty_cache()

        # Добавляем расчет PSNR и SSIM метрик (ВЫНЕСЛИ ИЗ БЛОКА ДИСКРИМИНАТОРА!)
        if current_iter % 100 == 0:  # Считаем каждые 100 итераций
            with torch.no_grad():
                # Ограничиваем выходные значения
                output_clamped = torch.clamp(self.output, 0, 1)

                # Конвертируем в numpy для расчета метрик
                try:
                    pred_np = tensor2img([output_clamped])  # Формат HWC, диапазон [0, 255]
                    gt_np = tensor2img([self.gt])  # Формат HWC, диапазон [0, 255]

                    # Рассчитываем PSNR
                    psnr_value = calculate_psnr(pred_np, gt_np, crop_border=0, test_y_channel=False)
                    loss_dict['psnr'] = psnr_value

                    # Рассчитываем SSIM
                    ssim_value = calculate_ssim(pred_np, gt_np, crop_border=0, test_y_channel=False)
                    loss_dict['ssim'] = ssim_value

                except Exception as e:
                    # В случае ошибки ставим нулевые значения
                    loss_dict['psnr'] = 0.0
                    loss_dict['ssim'] = 0.0
                    print(f"Ошибка расчета метрик: {e}")

        self.log_dict = self.reduce_loss_dict(loss_dict)
        if current_iter % 10 == 0:
            torch.cuda.empty_cache()

    def test(self):
        """Тестирование с сохранением физических свойств"""
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.lq)

            # Clamp для физической корректности (температуры не могут быть отрицательными в нормализованном виде)
            self.output = torch.clamp(self.output, 0, 1)

        self.net_g.train()

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """Валидация с метриками для температурных данных"""
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            self._initialize_best_metric_results(dataset_name)

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img

            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img

            # Сохранение изображений
            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             f'{current_iter}_{dataset_name}',
                                             f'{idx:08d}.png')
                else:
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             dataset_name,
                                             f'{idx:08d}.png')

                imwrite(sr_img, save_img_path)

            # Вычисление метрик
            if with_metrics:
                for name, opt_ in self.opt['val']['metrics'].items():
                    self._update_metric(metric_data, dataset_name, name, opt_)

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {idx:08d}')

        if use_pbar:
            pbar.close()

        if with_metrics:
            self._report_metric_results(dataset_name)

    def _initialize_best_metric_results(self, dataset_name):
        """Initialize metric results dict."""
        if not hasattr(self, 'best_metric_results'):
            self.best_metric_results = {}

        # Initialize record for this dataset
        record = {}
        for metric, content in self.opt['val']['metrics'].items():
            record[metric] = {'better': 'higher', 'val': float('-inf'), 'iter': -1}
            if content.get('better', 'higher') == 'lower':
                record[metric]['better'] = 'lower'
                record[metric]['val'] = float('inf')
        self.best_metric_results[dataset_name] = record

    def _update_metric(self, metric_data, dataset_name, metric_name, opt_):
        """Update metric results."""
        if metric_name == 'psnr':
            from basicsr.metrics import calculate_psnr
            value = calculate_psnr(metric_data['img'], metric_data['img2'],
                                   crop_border=opt_.get('crop_border', 0),
                                   test_y_channel=opt_.get('test_y_channel', False))
        elif metric_name == 'ssim':
            from basicsr.metrics import calculate_ssim
            value = calculate_ssim(metric_data['img'], metric_data['img2'],
                                   crop_border=opt_.get('crop_border', 0),
                                   test_y_channel=opt_.get('test_y_channel', False))
        else:
            # For other metrics, use basicsr's calculate_metric
            from basicsr.metrics import calculate_metric
            value = calculate_metric(metric_data, opt_)

        # Store the metric value
        if not hasattr(self, 'metric_results'):
            self.metric_results = {}
        if dataset_name not in self.metric_results:
            self.metric_results[dataset_name] = {}
        self.metric_results[dataset_name][metric_name] = value

        # Update best metric if needed
        if value > self.best_metric_results[dataset_name][metric_name]['val']:
            self.best_metric_results[dataset_name][metric_name]['val'] = value
            self.best_metric_results[dataset_name][metric_name]['iter'] = metric_data.get('iter', -1)

    def _report_metric_results(self, dataset_name):
        """Report average metrics with enhanced temperature-specific info."""
        from basicsr.utils import get_root_logger
        logger = get_root_logger()

        if hasattr(self, 'metric_results') and dataset_name in self.metric_results:
            logger.info(f'\n=== Результаты валидации для {dataset_name} ===')

            for metric_name, metric_value in self.metric_results[dataset_name].items():
                if metric_name == 'psnr':
                    logger.info(f'PSNR: {metric_value:.2f} dB')
                elif metric_name == 'ssim':
                    logger.info(f'SSIM: {metric_value:.4f}')
                else:
                    logger.info(f'{metric_name.upper()}: {metric_value:.4f}')

                # Report best metric
                if hasattr(self, 'best_metric_results') and dataset_name in self.best_metric_results:
                    best_info = self.best_metric_results[dataset_name][metric_name]
                    if metric_name == 'psnr':
                        logger.info(f'Лучший PSNR: {best_info["val"]:.2f} dB на итерации {best_info["iter"]}')
                    elif metric_name == 'ssim':
                        logger.info(f'Лучший SSIM: {best_info["val"]:.4f} на итерации {best_info["iter"]}')
                    else:
                        logger.info(f'Лучший {metric_name}: {best_info["val"]:.4f} на итерации {best_info["iter"]}')

            logger.info('=' * 60)