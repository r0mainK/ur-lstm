from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from models.utils import ModelType


def create_writer(log_dir: Path, model_type: ModelType, seed: int) -> SummaryWriter:
    log_dir = Path(__file__).parent / "runs" / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_dir = log_dir / f"{model_type}-{seed}_1"
    i = 1
    while log_dir.exists():
        i += 1
        log_dir = log_dir.parent / f"{model_type}-{seed}_{i}"
    return SummaryWriter(log_dir)
