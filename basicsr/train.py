import datetime
import logging
import math
import time
import torch
from os import path as osp

from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options

import os

enable_deepspeed = os.getenv('ENABLE_DEEPSPEED', 'false').lower() in ['true', '1']


def init_tb_loggers(opt):
    # initialize wandb logger before tensorboard logger to allow proper sync
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
    return tb_logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loaders = None, []
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.floor(
                len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info('Training statistics:'
                        f'\n\tNumber of train images: {len(train_set)}'
                        f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                        f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                        f'\n\tWorld size (gpu number): {opt["world_size"]}'
                        f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
            accum = opt['train'].get('accumulation_steps', 1)
            if accum > 1 and opt['train'].get('use_accumulation'):
                eff_batch = dataset_opt['batch_size_per_gpu'] * opt['world_size'] * accum
                steps_per_epoch = math.ceil(num_iter_per_epoch / accum)
                logger.info(f'\tGradient accumulation steps: {accum}')
                logger.info(f'\tEffective batch size per step: {eff_batch}')
                logger.info(f'\tParameter update steps per epoch: {steps_per_epoch}')
        elif phase.split('_')[0] == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
            val_loaders.append(val_loader)
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters


def load_resume_state(opt):
    resume_state_path = None
    if opt['auto_resume']:
        # state_path = osp.join('experiments', opt['name'], 'training_states')
        state_path = opt['path']['training_states']
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt['path']['resume_state'] = resume_state_path
    else:
        if opt['path'].get('resume_state'):
            resume_state_path = opt['path']['resume_state']
   
    if enable_deepspeed:
        return resume_state_path
    
    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
        check_resume(opt, resume_state['iter'])
    return resume_state

def auto_scale_iter_configs(opt, scale_num=1):
    train_opt = opt.get('train', {})
    if scale_num <= 1:
        logger = get_root_logger()
        logger.warning(f'[AutoScale] scale_num = 1 , return directly')
        return opt
    
    logger = get_root_logger()
    logger.info(f'[AutoScale] auto scale by {scale_num}')
    
    if 'total_iter' in train_opt:
        orig = train_opt['total_iter']
        train_opt['total_iter'] = int(orig * scale_num)
        logger.info(f'[AutoScale] train/total_iter: {orig} -> {train_opt["total_iter"]}')
    
    sched_opt = train_opt.get('scheduler', {})
    if 'milestones' in sched_opt:
        orig = sched_opt['milestones']
        sched_opt['milestones'] = [int(m * scale_num) for m in orig]
        logger.info(f'[AutoScale] train/scheduler/milestones: {orig} -> {sched_opt["milestones"]}')
    
    if 'warmup_iter' in train_opt:
        orig = train_opt['warmup_iter']
        train_opt['warmup_iter'] = int(orig * scale_num)
        logger.info(f'[AutoScale] train/warmup_iter: {orig} -> {train_opt["warmup_iter"]}')
    
    val_opt = opt.get('val', {})
    if 'val_freq' in val_opt:
        orig = val_opt['val_freq']
        val_opt['val_freq'] = int(orig * scale_num)
        logger.info(f'[AutoScale] val/val_freq: {orig} -> {val_opt["val_freq"]}')
    
    log_opt = opt.get('logger', {})
    if 'print_freq' in log_opt:
        orig = log_opt['print_freq']
        log_opt['print_freq'] = int(orig * scale_num)
        logger.info(f'[AutoScale] logger/print_freq: {orig} -> {log_opt["print_freq"]}')

    if 'save_checkpoint_freq' in log_opt:
        orig = log_opt['save_checkpoint_freq']
        log_opt['save_checkpoint_freq'] = int(orig * scale_num)
        logger.info(f'[AutoScale] logger/save_checkpoint_freq: {orig} -> {log_opt["save_checkpoint_freq"]}')
    
    opt['auto_scale_num'] = scale_num

    return opt

def train_pipeline(root_path):
    # parse options, set distributed setting, set random seed
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # load resume states if necessary
    resume_state = load_resume_state(opt)
    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

    # copy the yml file to the experiment root
    copy_opt_file(args.opt, opt['path']['experiments_root'])

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    log_level = logging.DEBUG if opt.get('debug', False) else logging.INFO 
    logger = get_root_logger(logger_name='basicsr', log_level=log_level, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    # initialize wandb and tb loggers
    tb_logger = init_tb_loggers(opt)
    # accumulation options change
    if opt['train'].get('use_accumulation'):
        auto_scale_iter_configs(opt, scale_num=opt['train'].get('accumulation_steps', 1))
    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

    # create model
    model = build_model(opt)
    model.tb_logger = tb_logger
    if enable_deepspeed:
        model.init_deepspeed(args)
        
        
    if enable_deepspeed:
        c_state = model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {c_state['epoch']}, iter: {c_state['iter']}.")
        start_epoch = c_state['epoch']
        current_iter = c_state['iter']
    elif resume_state:  # resume training
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        start_epoch = 0
        current_iter = 0
        
    

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)
    model.tb_logger = tb_logger

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f"Wrong prefetch_mode {prefetch_mode}. Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_timer.record()

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)
            iter_timer.record()
            if current_iter == 1:
                # reset start time in msg_logger for more accurate eta_time
                # not work in resume mode
                msg_logger.reset_start_time()
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            # validation
            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                if len(val_loaders) > 1:
                    logger.warning('Multiple validation datasets are *only* supported by SRModel.')
                for val_loader in val_loaders:
                    model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])

            data_timer.start()
            iter_timer.start()
            train_data = prefetcher.next()
            # end of one iter
        if opt['train'].get('use_accumulation'):
            model.flush_gradients(current_iter)
        # end of one epoch
    # end of epoches

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest

    if hasattr(model, 'remove_weight_gradient_hooks'):
        model.remove_weight_gradient_hooks()

    if opt.get('val') is not None:
        for val_loader in val_loaders:
            model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)