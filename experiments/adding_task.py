from pathlib import Path
import sys
import traceback

from omegaconf import OmegaConf
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import tqdm

from datasets.adding_dataset import AddingDataset
from models.adding_model import AddingModel
from models.utils import ModelType
from tensorboard_utils import create_writer


def main() -> None:
    conf = OmegaConf.load(Path(__file__).parent / "configs" / "adding_config.yaml")
    conf.model_type = ModelType(conf.model_type)
    writer = create_writer(Path(f"adding-task-{conf.seq_length}"), conf.model_type, conf.seed)
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    torch.random.manual_seed(conf.seed)
    model = AddingModel(conf.model_type, conf.hidden_size, conf.forget_bias)
    model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=conf.learning_rate)
    torch.random.manual_seed(conf.seed)
    dataset = AddingDataset(conf.seq_length)
    generator = DataLoader(dataset, batch_size=conf.batch_size)
    model.train()

    for step, (x, target) in tqdm.tqdm(enumerate(generator), desc="step", total=conf.n_steps):
        if step == conf.n_steps:
            break
        x = x.transpose(0, 1).to(device)
        target = target.view(conf.batch_size, 1).to(device)
        model.zero_grad()
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.lstm.parameters(), 1)
        optimizer.step()
        writer.add_scalar("MSE", loss.item(), step)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
