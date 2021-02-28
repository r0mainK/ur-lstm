from enum import Enum
from pathlib import Path
import sys
import traceback

from omegaconf import OmegaConf
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.datasets import MNIST
from torchvision.transforms import Compose
from torchvision.transforms import Lambda
from torchvision.transforms import ToTensor
import tqdm

from datasets.permuted_dataset import PermutatedDataset
from models.image_model import ImageModel
from models.utils import ModelType
from utils.tensorboard_utils import create_writer


class DatasetType(str, Enum):
    mnist = "mnist"
    permutated_mnist = "p-mnist"
    cifar_10 = "cifar-10"


def main() -> None:
    conf = OmegaConf.load(Path(__file__).parent / "configs" / "image_config.yaml")
    conf.model_type = ModelType(conf.model_type)
    conf.dataset_type = DatasetType(conf.model_type)
    writer = create_writer(
        Path(f"image-classification-{conf.dataset_type}"), conf.model_type, conf.seed
    )
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    torch.random.manual_seed(conf.seed)
    root_dir = Path(__file__).parent.parent / "data"

    if conf.dataset_type is DatasetType.cifar_10:
        n_channels = 3
        image_size = 32 * 32
        transform = Compose([ToTensor(), Lambda(lambda t: t.reshape(n_channels, -1).t())])
        train_dataset = CIFAR10(root_dir, download=True, train=True, transform=transform)
        test_dataset = CIFAR10(root_dir, download=True, train=False, transform=transform)
    else:
        n_channels = 1
        image_size = 28 * 28
        transform = Compose([ToTensor(), Lambda(lambda t: t.reshape(n_channels, -1).t())])
        train_dataset = MNIST(root_dir, download=True, train=True, transform=transform)
        test_dataset = MNIST(root_dir, download=True, train=False, transform=transform)
        if conf.dataset_type is DatasetType.permutated_mnist:
            train_dataset = PermutatedDataset(train_dataset, image_size)
            test_dataset = PermutatedDataset(test_dataset, image_size)

    model = ImageModel(
        conf.model_type,
        n_channels,
        image_size,
        conf.hidden_size_lstm,
        conf.hidden_size_relu,
        conf.forget_bias,
    )
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=conf.learning_rate)
    torch.random.manual_seed(conf.seed)
    train_generator = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True)
    test_generator = DataLoader(test_dataset, batch_size=conf.batch_size)
    step = 0

    for epoch in tqdm.tqdm(range(conf.n_epochs), desc="epoch"):
        model.train()
        for x, target in tqdm.tqdm(train_generator, desc="train batch", leave=False):
            x = x.transpose(0, 1).to(device)
            target = target.to(device)
            model.zero_grad()
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.lstm.parameters(), 1)
            optimizer.step()
            writer.add_scalar("Train Cross Entropy", loss.item(), step)
            step += 1
        model.eval()
        total_acc = 0
        total_loss = 0
        with torch.no_grad():
            for x, target in tqdm.tqdm(test_generator, desc="test batch", leave=False):
                x = x.transpose(0, 1).to(device)
                target = target.to(device)
                output = model(x)
                _, predicted = output.max(dim=1)
                total_acc += (predicted == target).sum().item()
                loss = loss_fn(output, target)
                total_loss += loss.item() * len(x[0])
        writer.add_scalar("Test Accuracy", total_acc / len(test_dataset), epoch)
        writer.add_scalar("Test Cross Entropy", total_loss / len(test_dataset), epoch)

    if conf.save_model:
        root_dir = Path(__file__).parent.parent / "saved_models" / conf.dataset_type
        root_dir.mkdir(parents=True, exist_ok=True)
        file_name = root_dir / f"{conf.model_type}-{conf.seed}.pt"
        torch.save(file_name, model.state_dict())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
