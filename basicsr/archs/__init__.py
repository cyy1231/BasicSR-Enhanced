import importlib
from copy import deepcopy
from os import path as osp
import os

from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import ARCH_REGISTRY

__all__ = ['build_network']

# automatically scan and import arch modules for registry
# scan all the files under the 'archs' folder and collect files ending with '_arch.py'
arch_folder = osp.dirname(osp.abspath(__file__))

arch_filenames = []
for root, dirs, files in os.walk(arch_folder):
    dirs[:] = [d for d in dirs if not d.startswith('_')]
    for f in files:
        if f.endswith('_arch.py'):
            rel_dir = osp.relpath(root, arch_folder).replace(os.sep, '.')
            module_name = osp.splitext(f)[0]
            if rel_dir == '.':
                arch_filenames.append(f'basicsr.archs.{module_name}')
            else:
                arch_filenames.append(f'basicsr.archs.{rel_dir}.{module_name}')

# auto import
for module_path in arch_filenames:
    try:
        importlib.import_module(module_path)
    except Exception as e:
        logger = get_root_logger()
        logger.warning(f'Failed to import {module_path}: {e}')


def build_network(opt):
    opt = deepcopy(opt)
    network_type = opt.pop('type')
    net = ARCH_REGISTRY.get(network_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Network [{net.__class__.__name__}] is created.')
    return net
