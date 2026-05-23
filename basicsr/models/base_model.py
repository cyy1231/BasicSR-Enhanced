import os
import time
import torch
from collections import OrderedDict
from copy import deepcopy
from torch.nn.parallel import DataParallel, DistributedDataParallel

from basicsr.models import lr_scheduler as lr_scheduler
from basicsr.utils import get_root_logger
from basicsr.utils.dist_util import master_only


class BaseModel():
    """Base model."""

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []
        self.tb_logger = None
        self.checkpoint_manager = None

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def save(self, epoch, current_iter):
        """Save networks and training state."""
        pass

    def validation(self, dataloader, current_iter, tb_logger, save_img=False):
        """Validation function.

        Args:
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
            current_iter (int): Current iteration.
            tb_logger (tensorboard logger): Tensorboard logger.
            save_img (bool): Whether to save images. Default: False.
        """
        if self.opt['dist']:
            self.dist_validation(dataloader, current_iter, tb_logger, save_img)
        else:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

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

    def model_ema(self, decay=0.999):
        net_g = self.get_bare_model(self.net_g)

        net_g_params = dict(net_g.named_parameters())
        net_g_ema_params = dict(self.net_g_ema.named_parameters())

        for k in net_g_ema_params.keys():
            net_g_ema_params[k].data.mul_(decay).add_(net_g_params[k].data, alpha=1 - decay)

    def get_current_log(self):
        return self.log_dict

    def model_to_device(self, net):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.

        Args:
            net (nn.Module)
        """
        net = net.to(self.device)
        if self.opt['dist']:
            find_unused_parameters = self.opt.get('find_unused_parameters', False)
            net = DistributedDataParallel(
                net, device_ids=[torch.cuda.current_device()], find_unused_parameters=find_unused_parameters)
        elif self.opt['num_gpu'] > 1:
            net = DataParallel(net)
        return net

    def get_optimizer(self, optim_type, params, lr, **kwargs):
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        elif optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr, **kwargs)
        elif optim_type == 'Adamax':
            optimizer = torch.optim.Adamax(params, lr, **kwargs)
        elif optim_type == 'SGD':
            optimizer = torch.optim.SGD(params, lr, **kwargs)
        elif optim_type == 'ASGD':
            optimizer = torch.optim.ASGD(params, lr, **kwargs)
        elif optim_type == 'RMSprop':
            optimizer = torch.optim.RMSprop(params, lr, **kwargs)
        elif optim_type == 'Rprop':
            optimizer = torch.optim.Rprop(params, lr, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supported yet.')
        return optimizer

    def setup_schedulers(self):
        """Set up schedulers."""
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.MultiStepRestartLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartLR':
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.CosineAnnealingRestartLR(optimizer, **train_opt['scheduler']))
        else:
            raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented yet.')

    def get_bare_model(self, net):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net

    @master_only
    def print_network(self, net):
        """Print the str and parameter number of a network.

        Args:
            net (nn.Module)
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net_cls_str = f'{net.__class__.__name__} - {net.module.__class__.__name__}'
        else:
            net_cls_str = f'{net.__class__.__name__}'

        net = self.get_bare_model(net)
        net_params = sum(map(lambda x: x.numel(), net.parameters()))

        logger = get_root_logger()
        logger.info(f'Network: {net_cls_str}, with parameters: {net_params:,d}')

    @master_only
    def save_network_architecture(self, net, save_name='network_arch.txt',
                                   exporter_fn=None, fmt=None):
        """Save layer-by-layer network architecture to a dedicated file.

        Supports multiple output formats: txt, json, md, csv.
        Users may also provide a custom exporter_fn.

        Args:
            net (nn.Module): Network to analyze.
            save_name (str): Output filename. The extension determines the
                format if fmt is not specified (.txt, .json, .md, .csv).
            exporter_fn (callable, optional): Custom exporter function.
                Signature: fn(net, save_path, net_cls_name, device).
                If provided, fmt is ignored.
            fmt (str, optional): Output format, overriding extension detection.
                Options: 'txt', 'json', 'md', 'csv'.

        Raises:
            ValueError: If fmt is unknown and no exporter_fn is provided.
        """
        from basicsr.utils.arch_exporter import get_exporter, export_arch_text

        net = self.get_bare_model(net)

        # Determine which exporter to use
        if exporter_fn is not None:
            # User-provided custom exporter
            pass
        else:
            if fmt is None:
                # Infer format from file extension
                ext = os.path.splitext(save_name)[1].lstrip('.').lower()
                fmt = ext if ext in ('json', 'md', 'csv') else 'txt'
            exporter_fn = get_exporter(fmt)

        save_path = os.path.join(
            self.opt['path']['experiments_root'], save_name
        )
        logger = get_root_logger()

        # Extract network metadata
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net_cls_str = (
                f'{net.__class__.__name__} - '
                f'{net.module.__class__.__name__}'
            )
        else:
            net_cls_str = net.__class__.__name__

        device = next(net.parameters()).device

        # Execute export
        exporter_fn(net, save_path, net_cls_str, device)

        # Log summary
        total_params = sum(p.numel() for p in net.parameters())
        logger.info(
            f'Network architecture saved to {save_path} '
            f'(Format: {fmt or "custom"}, Total params: {total_params:,})'
        )

    @master_only
    def save_network_complexity(self, net, save_name='network_complexity.txt'):
        """Save network complexity (FLOPs / MACs) to a dedicated text file.
        
        Input resolution is automatically inferred from opt:
          1. datasets.train.lq_size (if explicitly set)
          2. datasets.train.gt_size // scale (fallback)
        """
        net = self.get_bare_model(net)
        save_path = os.path.join(self.opt['path']['experiments_root'], save_name)
        logger = get_root_logger()

        # Infer input resolution from opt
        train_opt = self.opt.get('datasets', {}).get('train', {})

        def _parse_size(size_val, scale=1):
            if size_val is None:
                return None
            if isinstance(size_val, int):
                return (size_val // scale, size_val // scale)
            if isinstance(size_val, (list, tuple)) and len(size_val) >= 2:
                return (size_val[0] // scale, size_val[1] // scale)
            return None
        
        lq_size = _parse_size(train_opt.get('lq_size'))
        
        if lq_size is None:
            gt_size = train_opt.get('gt_size', 128)
            scale = self.opt.get('scale', 1)
            lq_size = _parse_size(gt_size, scale=scale)
        
        input_h, input_w = lq_size
        in_chans = self.opt.get('network_g', {}).get('num_in_ch', 3)
        input_res = (in_chans, input_h, input_w)  # (C, H, W) for ptflops


        # log write
        lines = []
        lines.append('=' * 60)
        lines.append(f'Network Class: {net.__class__.__name__}')
        lines.append(f'Input Resolution (LR): {input_h}x{input_w}')
        lines.append('=' * 60)

        total_params = sum(p.numel() for p in net.parameters())
        lines.append(f'Total parameters:      {total_params:,}')
        lines.append('')

        try:
            from ptflops import get_model_complexity_info
            in_chans = self.opt.get('network_g', {}).get('num_in_ch', 3)
            macs, params = get_model_complexity_info(
                net, (in_chans, input_h, input_w), as_strings=True,
                print_per_layer_stat=False, verbose=False
            )
            lines.append(f'MACs (GMac):           {macs}')
            lines.append(f'Params (ptflops):      {params}')
        except ImportError:
            lines.append('ptflops not installed, skipping MACs calculation.')
            macs = params = 'N/A'
        except Exception as e:
            lines.append(f'Complexity calculation failed: {e}')
            macs = params = 'N/A'

        lines.append('=' * 60)

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        logger.info(
            f'Network complexity saved to {save_path} '
            f'(Input: {input_h}x{input_w}, MACs: {macs})'
        )

    def _set_lr(self, lr_groups_l):
        """Set learning rate for warm-up.

        Args:
            lr_groups_l (list): List for lr_groups, each for an optimizer.
        """
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        """Get the initial lr, which is set by the scheduler.
        """
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, current_iter, warmup_iter=-1):
        """Update learning rate.

        Args:
            current_iter (int): Current iteration.
            warmup_iter (int)： Warm-up iter numbers. -1 for no warm-up.
                Default： -1.
        """
        if current_iter > 1:
            for scheduler in self.schedulers:
                scheduler.step()
        # set up warm-up learning rate
        if current_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            # currently only support linearly warm up
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * current_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        return [param_group['lr'] for param_group in self.optimizers[0].param_groups]

    @master_only
    def save_network(self, net, net_label, current_iter, param_key='params'):
        """Save networks.

        Args:
            net (nn.Module | list[nn.Module]): Network(s) to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
            param_key (str | list[str]): The parameter key(s) to save network.
                Default: 'params'.
        """
        if current_iter == -1:
            current_iter = 'latest'
        save_filename = f'{net_label}_{current_iter}.pth'
        save_path = os.path.join(self.opt['path']['models'], save_filename)

        net = net if isinstance(net, list) else [net]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(param_key), 'The lengths of net and param_key should be the same.'

        save_dict = {}
        for net_, param_key_ in zip(net, param_key):
            net_ = self.get_bare_model(net_)
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                if key.startswith('module.'):  # remove unnecessary 'module.'
                    key = key[7:]
                state_dict[key] = param.cpu()
            save_dict[param_key_] = state_dict

        # avoid occasional writing errors
        retry = 3
        while retry > 0:
            try:
                torch.save(save_dict, save_path)
            except Exception as e:
                logger = get_root_logger()
                logger.warning(f'Save model error: {e}, remaining retry times: {retry - 1}')
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:
            logger.warning(f'Still cannot save {save_path}. Just ignore it.')
            # raise IOError(f'Cannot save {save_path}.')

    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        """Print keys with different name or different size when loading models.

        1. Print keys with different names.
        2. If strict=False, print the same key but with different tensor size.
            It also ignore these keys with different sizes (not load).

        Args:
            crt_net (torch model): Current network.
            load_net (dict): Loaded network.
            strict (bool): Whether strictly loaded. Default: True.
        """
        crt_net = self.get_bare_model(crt_net)
        crt_net = crt_net.state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        logger = get_root_logger()
        if crt_net_keys != load_net_keys:
            logger.warning('Current net - loaded net:')
            for v in sorted(list(crt_net_keys - load_net_keys)):
                logger.warning(f'  {v}')
            logger.warning('Loaded net - current net:')
            for v in sorted(list(load_net_keys - crt_net_keys)):
                logger.warning(f'  {v}')

        # check the size for the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    logger.warning(f'Size different, ignore [{k}]: crt_net: '
                                   f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)

    def load_network(self, net, load_path, strict=True, param_key='params'):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        logger = get_root_logger()
        net = self.get_bare_model(net)
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                logger.info('Loading: params_ema does not exist, use params.')
            load_net = load_net[param_key]
        logger.info(f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)

    @master_only
    def save_training_state(self, epoch, current_iter):
        """Save training states during training, which will be used for
        resuming.

        Args:
            epoch (int): Current epoch.
            current_iter (int): Current iteration.
        """
        if current_iter != -1:
            state = {'epoch': epoch, 'iter': current_iter, 'optimizers': [], 'schedulers': []}
            for o in self.optimizers:
                state['optimizers'].append(o.state_dict())
            for s in self.schedulers:
                state['schedulers'].append(s.state_dict())
            save_filename = f'{current_iter}.state'
            save_path = os.path.join(self.opt['path']['training_states'], save_filename)

            # avoid occasional writing errors
            retry = 3
            while retry > 0:
                try:
                    torch.save(state, save_path)
                except Exception as e:
                    logger = get_root_logger()
                    logger.warning(f'Save training state error: {e}, remaining retry times: {retry - 1}')
                    time.sleep(1)
                else:
                    break
                finally:
                    retry -= 1
            if retry == 0:
                logger.warning(f'Still cannot save {save_path}. Just ignore it.')
                # raise IOError(f'Cannot save {save_path}.')

    def resume_training(self, resume_state):
        """Reload the optimizers and schedulers for resumed training.

        Args:
            resume_state (dict): Resume state.
        """
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

    def reduce_loss_dict(self, loss_dict):
        """reduce loss dict.

        In distributed training, it averages the losses among different GPUs .

        Args:
            loss_dict (OrderedDict): Loss dict.
        """
        with torch.no_grad():
            if self.opt['dist']:
                keys = []
                losses = []
                for name, value in loss_dict.items():
                    keys.append(name)
                    losses.append(value)
                losses = torch.stack(losses, 0)
                torch.distributed.reduce(losses, dst=0)
                if self.opt['rank'] == 0:
                    losses /= self.opt['world_size']
                loss_dict = {key: loss for key, loss in zip(keys, losses)}

            log_dict = OrderedDict()
            for name, value in loss_dict.items():
                log_dict[name] = value.mean().item()

            return log_dict
        

    def _init_gradient_accumulation(self):
        """Initialize gradient accumulation settings."""
        if self.opt['train'].get('use_accumulation'):
            train_opt = self.opt.get('train', {})
            self.accumulation_steps = train_opt.get('accumulation_steps', 1)
            self._accumulation_counter = 0
            if self.accumulation_steps > 1:
                logger = get_root_logger()
                logger.info(f'Use gradient accumulation with steps: {self.accumulation_steps}')
        else: 
            if self.opt.get('auto_scale_num'):
                logger = get_root_logger()
                logger.info(f'Use AutoScale but not use Accumulation')

    def _accumulation_step_begin(self):
        """Call at the very beginning of optimize_parameters().
        
        Handles counter increment and zero_grad (only on first step of window).
        """
        if self.opt['train'].get('use_accumulation'):
            self._accumulation_counter += 1
            self._is_first_accum_step = (self._accumulation_counter == 1)
            self._is_last_accum_step = (self._accumulation_counter == self.accumulation_steps)
            if self._is_first_accum_step:
                for optimizer in self.optimizers:
                    optimizer.zero_grad()
        else:
            for optimizer in self.optimizers:
                optimizer.zero_grad()

    def _scale_loss_for_accumulation(self, loss):
        """Scale loss before backward(). No-op if accumulation is disabled."""
        if hasattr(self, 'accumulation_steps') and self.accumulation_steps > 1:
            return loss / self.accumulation_steps
        return loss

    def _accumulation_step_end(self, loss_dict, current_iter=None):
        """Call at the very end of optimize_parameters().
        
        Handles step(), EMA, log_dict assembly, and accumulation metadata.
        Returns the assembled log_dict.
        """
        if self.opt['train'].get('use_accumulation'):
            if self._is_last_accum_step:
                for optimizer in self.optimizers:
                    optimizer.step()
                self._accumulation_counter = 0

            # distributed loss reduction + metadata
            log_dict = self.reduce_loss_dict(loss_dict)

            if self.accumulation_steps > 1:
                current_accum = self.accumulation_steps if self._is_last_accum_step else self._accumulation_counter
                log_dict['grad_accum'] = current_accum
                log_dict['accum_total'] = self.accumulation_steps

            # EMA update only when parameters actually changed
            if self._is_last_accum_step and getattr(self, 'ema_decay', 0) > 0:
                self.model_ema(decay=self.ema_decay)
            
            # if weight visual, weight visual to tb_logger
            if (current_iter is not None
            and hasattr(self, 'tb_logger') 
            and self.tb_logger is not None
            and hasattr(self, '_weight_log_handles') 
            and self._weight_log_handles
            and self._is_last_accum_step):
                for net_name in self._weight_log_handles.keys():
                    target_net = self._weight_log_configs[net_name].get('instance')        
                    if target_net is not None:
                        self.log_weight_gradient_distributions(
                            target_net, net_name=net_name,
                            tb_logger=self.tb_logger, current_iter=current_iter,
                            prefix='train'
                        )

        else:
            for optimizer in self.optimizers:
                optimizer.step()
            # distributed loss reduction + metadata
            log_dict = self.reduce_loss_dict(loss_dict)

            # EMA update only when parameters actually changed
            if getattr(self, 'ema_decay', 0) > 0:
                self.model_ema(decay=self.ema_decay)

            # if weight visual, weight visual to tb_logger
            if (current_iter is not None
            and hasattr(self, 'tb_logger') 
            and self.tb_logger is not None
            and hasattr(self, '_weight_log_handles') 
            and self._weight_log_handles):
                for net_name in self._weight_log_handles.keys():
                    target_net = self._weight_log_configs[net_name].get('instance')
                    if target_net is not None:
                        self.log_weight_gradient_distributions(
                            target_net, net_name=net_name,
                            tb_logger=self.tb_logger, current_iter=current_iter,
                            prefix='train'
                        )

        return log_dict

    def flush_gradients(self, current_iter=-1):
        """Force a parameter update for any remaining accumulated gradients.
        Call this before breaking out of the training loop or at epoch end.
        """
        if getattr(self, '_accumulation_counter', 0) > 0:
            logger = get_root_logger()
            logger.info(
                f'Flush remaining {self._accumulation_counter} accumulated gradients '
                f'at iter {current_iter}'
            )
            for optimizer in self.optimizers:
                optimizer.step()
            self._accumulation_counter = 0
            if getattr(self, 'ema_decay', 0) > 0:
                self.model_ema(decay=self.ema_decay)


    def setup_feature_hooks(self, target_net: torch.nn.Module, net_name='Net', layer_names=None):
        """Register forward hooks to capture intermediate features.
        
        Args:
            target_net (nn.Module): Network to hook on.
            net_name (str): Network identifier, used for logging and filename.
                Should match the 'name' in visual_config.
            layer_names (list[str]): Names of layers to hook. 
                If None, read from opt['visualize_layers'].
        """
        # defensive
        if target_net is None:
            logger = get_root_logger()
            logger.warning(f'[Hook-{net_name}] target_net is None, skip feature hooks')
            return
        
        if not layer_names:
            logger = get_root_logger()
            logger.warning(f'[Hook-{net_name}] layer_names is empty, skip feature hooks')
            return

        # init
        if not hasattr(self, '_captured_features'):
            self._captured_features = {}
        if not hasattr(self, '_feature_hook_handles'):
            self._feature_hook_handles = {}
        
        self._captured_features[net_name] = {}
        self._feature_hook_handles[net_name] = []
        
        logger = get_root_logger()

        def make_hook(name):
            def hook(module, input, output):
                feat = output[0] if isinstance(output, tuple) else output
                self._captured_features[net_name][name] = feat.detach().cpu()
            return hook

        for name, module in target_net.named_modules():
            if name in layer_names:
                handle = module.register_forward_hook(make_hook(name))
                self._feature_hook_handles[net_name].append(handle)
                logger.debug(f'[Hook-{net_name}] Registered feature capture: {name}')

    def remove_feature_hooks(self, net_name=None):
        """Remove forward hooks.
        
        Args:
            net_name (str): Remove hooks for specific network.
                If None, remove all hooks for all networks.
        """
        if not hasattr(self, '_feature_hook_handles'):
            return
        
        if net_name is not None:
            handles = self._feature_hook_handles.get(net_name, [])
            for handle in handles:
                handle.remove()
            self._feature_hook_handles[net_name] = []
            if hasattr(self, '_captured_features') and net_name in self._captured_features:
                self._captured_features[net_name].clear()
            
            num_hooks = len(handles)
            if num_hooks > 0:
                logger = get_root_logger()
                logger.debug(f'[Hook-{net_name}] Removed {num_hooks} feature capture hook(s)')
        else:
            # all net remove
            for name, handles in self._feature_hook_handles.items():
                for handle in handles:
                    handle.remove()
                if hasattr(self, '_captured_features') and name in self._captured_features:
                    self._captured_features[name].clear()
            
            total = sum(len(h) for h in self._feature_hook_handles.values())
            self._feature_hook_handles = {}
            if total > 0:
                logger = get_root_logger()
                logger.debug(f'[Hook] Removed {total} feature capture hook(s) total')

    def save_captured_features(self, save_dir, net_name='net', prefix='', 
                               visualizer_fn=None):
        """Save captured intermediate features using an external visualizer.
        
        Args:
            save_dir (str): Directory to save images.
            net_name (str): Which network's features to save.
            prefix (str): Filename prefix.
            visualizer_fn (callable): Function to process and save each feature tensor.
                Signature: fn(tensor, save_path, **vis_kwargs).
                If None, uses default_feature_visualizer from basicsr.utils.visualize.
            **vis_kwargs: Additional arguments passed to visualizer_fn.
        """
        from basicsr.utils.visualize import default_feature_visualizer, save_rgb_image
        
        if not hasattr(self, '_captured_features'):
            return
        
        features = self._captured_features.get(net_name, {})
        if not features:
            return

        if visualizer_fn is None:
            visualizer_fn = default_feature_visualizer

        os.makedirs(save_dir, exist_ok=True)

        for layer_name, tensor in features.items():
            if tensor.dim() != 4:
                continue

            feat = tensor[0]  # (C, H, W)
            C, H, W = feat.shape

            fname = f"{prefix}_{net_name}_{layer_name.replace('.', '_')}.png"
            save_path = os.path.join(save_dir, fname)

            if C == 3:
                save_rgb_image(feat, save_path)
            else:
                default_feature_visualizer(tensor, save_path)
        
        self._captured_features[net_name].clear()

    def setup_weight_gradient_hooks(self, target_net: torch.nn.Module, net_name='Net', 
                                    layer_patterns=None, log_freq=500):
        """Register hooks to log weight and gradient distributions to TensorBoard.
        
        Args:
            target_net (nn.Module): Network to hook on.
            net_name (str): Network identifier, used for logging and filename.
            layer_patterns (list[str]): Substrings to match layer names.
            log_freq (int): Log every N parameter update steps.
        """
        if target_net is None:
            logger = get_root_logger()
            logger.warning(f'[WeightHook-{net_name}] target_net is None, skip')
            return
        if layer_patterns is None:
            logger = get_root_logger()
            logger.warning(f'[WeightHook-{net_name}] layer_patterns is empty, skip')
            return
        
        if not self.opt['logger'].get('use_tb_logger', False) or not hasattr(self, 'tb_logger'):
            return

        if not hasattr(self, '_weight_log_configs'):
            self._weight_log_configs = {}
        self._weight_log_configs[net_name] = {
            'instance': target_net,  # net instance
            'patterns': layer_patterns or [''],
            'freq': log_freq,
            'counter': 0
        }
        
        if not hasattr(self, '_weight_log_handles'):
            self._weight_log_handles = {}
        if not hasattr(self, '_captured_gradients'):
            self._captured_gradients = {}
        
        self._weight_log_handles[net_name] = []
        self._captured_gradients[net_name] = {}

        logger = get_root_logger()
        logger.info(f'[WeightHook-{net_name}] Setup weight/gradient logging every {log_freq} steps')

        def make_grad_hook(name):
            def hook(module, grad_input, grad_output):
                grad = grad_output[0] if isinstance(grad_output, tuple) else grad_output
                if grad is not None:
                    self._captured_gradients[net_name][name] = grad.detach()
            return hook

        target_net = self.get_bare_model(target_net)
        if target_net is None:
            return

        patterns = self._weight_log_configs[net_name]['patterns']
        for name, module in target_net.named_modules():
            has_params = any(p.requires_grad for p in module.parameters(recurse=False))
            if not has_params:
                continue
            if not any(pat in name for pat in patterns):
                continue

            handle = module.register_full_backward_hook(make_grad_hook(name))
            self._weight_log_handles[net_name].append(handle)

    def remove_weight_gradient_hooks(self, net_name=None):
        """Remove backward hooks.
        
        Args:
            net_name (str): Remove hooks for specific network.
                If None, remove all hooks for all networks.
        """
        if not hasattr(self, '_weight_log_handles'):
            return
        
        if net_name is not None:
            handles = self._weight_log_handles.get(net_name, [])
            for handle in handles:
                handle.remove()
            self._weight_log_handles[net_name] = []
            
            if hasattr(self, '_captured_gradients') and net_name in self._captured_gradients:
                self._captured_gradients[net_name].clear()
            
            num_hooks = len(handles)
            if num_hooks > 0:
                logger = get_root_logger()
                logger.info(f'[WeightHook-{net_name}] Removed {num_hooks} hook(s)')
        else:
            total = 0
            for name, handles in self._weight_log_handles.items():
                for handle in handles:
                    handle.remove()
                total += len(handles)
                if hasattr(self, '_captured_gradients') and name in self._captured_gradients:
                    self._captured_gradients[name].clear()
            
            self._weight_log_handles = {}
            if total > 0:
                logger = get_root_logger()
                logger.info(f'[WeightHook] Removed {total} hook(s) total')

    def log_weight_gradient_distributions(self, target_net: torch.nn.Module, net_name='net',
                                          tb_logger=None, current_iter=None, prefix='train'):
        """Log captured weight and gradient distributions to TensorBoard.
        
        Args:
            target_net (nn.Module): Network to read weights from.
            net_name (str): Which network's gradients to log.
            tb_logger: TensorBoard logger.
            current_iter (int): Current iteration.
            prefix (str): Tag prefix.
        """
        if target_net is None or not hasattr(self, '_captured_gradients') or not hasattr(self, '_weight_log_configs'):
            return
    
        grads = self._captured_gradients.get(net_name, {})
        if not grads or net_name not in self._weight_log_configs:
            return
        
        config = self._weight_log_configs[net_name]
        config['counter'] += 1
        
        if config['counter'] % config['freq'] != 0:
            grads.clear()
            return
        
        if tb_logger is None:
            grads.clear()
            return
        
        target_net = self.get_bare_model(target_net)

        for name, module in target_net.named_modules():
            if name not in grads:
                continue

            # Weight distribution
            params = list(module.parameters(recurse=False))
            if params:
                w = params[0].detach()
                tag_w = f'{prefix}/{net_name}/weight/{name.replace(".", "/")}'
                tb_logger.add_histogram(tag_w, w, current_iter)

            # Gradient distribution
            g = grads[name]
            if g is not None:
                tag_g = f'{prefix}/{net_name}/grad/{name.replace(".", "/")}'
                tb_logger.add_histogram(tag_g, g, current_iter)

        grads.clear()





