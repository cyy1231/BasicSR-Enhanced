import os

from basicsr.utils.logger import get_root_logger


class CheckpointManager:
    """Keep only the most recent N checkpoints to save disk space."""
    
    def __init__(self, max_keep=5):
        self.max_keep = max_keep
        self.saved_paths = []  # FIFO queue of (pth_path, state_path)
    
    def register(self, pth_path, state_path=None):
        self.saved_paths.append((pth_path, state_path))
        self._cleanup()
    
    def _cleanup(self):
        while len(self.saved_paths) > self.max_keep:
            old_pth, old_state = self.saved_paths.pop(0)
            for path in [old_pth, old_state]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                        logger = get_root_logger()
                        logger.info(f'[CheckpointManager] Removed old checkpoint: {path}')
                    except OSError as e:
                        logger = get_root_logger()
                        logger.warning(f'[CheckpointManager] Failed to remove {path}: {e}')