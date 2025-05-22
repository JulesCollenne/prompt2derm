import os
import logging
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# -------------------------------
# Console + File Logger Setup
# -------------------------------
def setup_logger(log_file=None, name="prompt2derm"):
    """
    Sets up a logger that logs to both console and file.

    Args:
        log_file (str): Path to a log file (optional).
        name (str): Name of the logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate logs if reimported
    if logger.handlers:
        return logger

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)

    return logger

# -------------------------------
# TensorBoard Logger
# -------------------------------
def setup_tensorboard(log_dir):
    """
    Returns a TensorBoard SummaryWriter.
    """
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir=log_dir)

# -------------------------------
# Weights & Biases Logger
# -------------------------------
def setup_wandb(project_name="prompt2derm", config=None, enable=True):
    """
    Initializes Weights & Biases (optional).

    Args:
        project_name (str): wandb project name
        config (dict): configuration to log
        enable (bool): whether to initialize wandb (fallback if disabled)

    Returns:
        run: wandb run object (or None if disabled)
    """
    if not WANDB_AVAILABLE or not enable:
        return None

    wandb.login()
    run = wandb.init(project=project_name, config=config)
    return run
