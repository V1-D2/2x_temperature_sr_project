# Конфигурация для обучения температурной Super-Resolution модели

# Общие параметры
name = 'TemperatureSR_SwinIR_ESRGAN_x2'
model_type = 'TemperatureSRModel'
scale = 2
num_gpu = 1  # Количество GPU

# Параметры данных
datasets = {
    'train': {
        'name': 'TemperatureTrainDataset',
        'dataroot_gt': None,  # Будет задан в train script
        'npz_files': [],  # Будет задан в train script
        'preprocessor_args': {
            'target_height': 2000,
            'target_width': 220
        },
        'scale_factor': 2,
        'batch_size': 16,
        'samples_per_file': 10000,  # Ограничение для управления памятью
        'num_worker': 4,
        'pin_memory': True,
        'persistent_workers': True
    },
    'val': {
        'name': 'TemperatureValDataset',
        'dataroot_gt': None,
        'npz_file': None,  # Будет задан в train script
        'n_samples': 10,
        'scale_factor': 2
    }
}

# Параметры сети
network_g = {
    'type': 'SwinIR',
    'upscale': 2,
    'in_chans': 1,  # Температурные данные - 1 канал
    'img_size': 64,
    'window_size': 8,
    'img_range': 1.,
    'depths': [8, 8, 8, 8, 8, 8, 8, 8],
    'embed_dim': 240,
    'num_heads': [8, 8, 8, 8, 8, 8, 8, 8],
    'mlp_ratio': 4,
    'upsampler': 'pixelshuffle',
    'resi_connection': '3conv'
}

network_d = {
    'type': 'UNetDiscriminatorSN',
    'num_in_ch': 1,  # Температурные данные - 1 канал
    'num_feat': 64,
    'skip_connection': True
}

# Путь к файлам
path = {
    'pretrain_network_g': None,
    'strict_load_g': True,
    'resume_state': None,
    'root': './',
    'experiments_root': './experiments',
    'models': './experiments/models',
    'training_states': './experiments/training_states',
    'log': './experiments/log',
    'visualization': './experiments/visualization'
}

# Параметры обучения
train = {
    'ema_decay': 0.999,
    'optim_g': {
        'type': 'Adam',
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'betas': [0.9, 0.99]
    },
    'optim_d': {
        'type': 'Adam',
        'lr': 5e-5,
        'weight_decay': 1e-4,
        'betas': [0.9, 0.99]
    },
    'scheduler': {
        'type': 'CosineAnnealingLR',
        'T_max': 100000,
        'eta_min': 1e-7
    },
    # Loss функции
    'pixel_opt': {
        'type': 'PhysicsConsistencyLoss',
        'loss_weight': 1.0,
        'gradient_weight': 0.08,
        'smoothness_weight': 0.03,
        'reduction': 'mean'
    },
    'perceptual_opt': {
        'type': 'TemperaturePerceptualLoss',
        'loss_weight': 0.1,
        'feature_weights': [0.1, 0.2, 1.0, 1.0]
    },
    'gan_opt': {
        'type': 'gan',
        'gan_type': 'lsgan',
        'real_label_val': 1.0,
        'fake_label_val': 0.0,
        'loss_weight': 0.1
    },
    # Параметры дискриминатора
    'net_d_iters': 1,
    'net_d_init_iters': 1000,
    # Частота сохранения
    'manual_seed': 10,
    'use_grad_clip': True,
    'grad_clip_norm': 0.1,
    'use_ema': True                 # Exponential Moving Averag
}

# Параметры валидации
val = {
    'val_freq': 10000,
    'save_img': True,
    'metrics': {
        'psnr': {
            'type': 'calculate_psnr',
            'crop_border': 0,
            'test_y_channel': False
        },
        'ssim': {
            'type': 'calculate_ssim',
            'crop_border': 0,
            'test_y_channel': False
        }
    }
}

# Логирование
logger = {
    'print_freq': 10000,
    'save_checkpoint_freq': 100000,
    'use_tb_logger': True,
    'wandb': {
        'project': 'temperature-sr',
        'resume_id': None
    }
}

# Распределенное обучение
dist_params = {
    'backend': 'nccl',
    'port': 29500
}

# Специфичные для температурных данных параметры
temperature_specific = {
    'preserve_relative_values': True,
    'temperature_range': [80, 400],  # Кельвины
    'physical_constraints': {
        'enforce_smoothness': True,
        'preserve_gradients': True,
        'max_gradient': 10.0  # Максимальный градиент температуры
    }
}

# Инкрементальное обучение
incremental_training = {
    'enabled': True,
    'epochs_per_file': 1,
    'learning_rate_decay_per_file': 1.0,
    'checkpoint_per_file': False,
    'shuffle_files': True
}

# Дополнительные параметры
others = {
    'use_amp': False,  # Automatic Mixed Precision
    'num_threads': 8,
    'seed': 10
}