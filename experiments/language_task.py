import math
from pathlib import Path
import sys
import traceback
import warnings

from omegaconf import OmegaConf
import torch
import torch.nn as nn
from torch.optim import AdamW
from torchtext.data import BPTTIterator
from torchtext.datasets import WikiText103
import tqdm

from models.language_model import LanguageModel
from models.utils import ModelType
from tensorboard_utils import create_writer


# TODO: Remove this when new torchtext API is released and integrated
warnings.filterwarnings("ignore", category=UserWarning)


def main() -> None:
    conf = OmegaConf.load(Path(__file__).parent / "configs" / "language_config.yaml")
    conf.model_type = ModelType(conf.model_type)
    writer = create_writer(Path("language-modeling"), conf.model_type, conf.seed)
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    torch.random.manual_seed(conf.seed)
    root_dir = Path(__file__).parent.parent / "data"
    WikiText103.download(root_dir)
    train_generator, val_generator, test_generator = WikiText103.iters(
        batch_size=conf.batch_size, bptt_len=conf.seq_length, root=root_dir, device=device
    )

    model = LanguageModel(
        conf.model_type,
        len(train_generator.dataset.fields["text"].vocab),
        conf.embedding_size,
        conf.hidden_size,
        conf.dropout_rate,
        conf.forget_bias,
    )
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=conf.learning_rate)
    step = 0

    for epoch in tqdm.tqdm(range(conf.n_epochs), desc="epoch"):
        model.train()
        for batch in tqdm.tqdm(train_generator, desc="train batch", leave=False):
            model.zero_grad()
            optimizer.zero_grad()
            output, _ = model(batch.text, None)
            loss = loss_fn(output.view(-1, output.size(-1)), batch.target.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.lstm.parameters(), 1)
            optimizer.step()
            writer.add_scalar("Train Cross Entropy", loss.item(), step)
            step += 1
        total_loss = evaluate(model, val_generator, loss_fn)
        writer.add_scalar("Validation Cross Entropy", total_loss, epoch)
        writer.add_scalar("Validation Perplexity", math.exp(total_loss), epoch)

    total_loss = evaluate(model, test_generator, loss_fn)
    print(f"test loss: {total_loss}")
    print(f"test perplexity: {math.exp(total_loss)}")

    if conf.save_model:
        root_dir = Path(__file__).parent.parent / "saved_models" / conf.dataset_type
        root_dir.mkdir(parents=True, exist_ok=True)
        file_name = root_dir / f"{conf.model_type}-{conf.seed}.pt"
        torch.save(file_name, model.state_dict())
        file_name = root_dir / f"{conf.model_type}-{conf.seed}-vocab.pt"
        torch.save(file_name, train_generator.dataset.fields["text"].vocab.itos)


def evaluate(model: nn.Module, generator: BPTTIterator, loss_fn: nn.CrossEntropyLoss):
    model.eval()
    total_loss = 0
    n = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(generator, desc="eval batch", leave=False):
            output, _ = model(batch.text, None)
            loss = loss_fn(output.view(-1, output.size(-1)), batch.target.view(-1))
            total_loss += loss.item() * len(batch.text)
            n += len(batch.text)
    total_loss = total_loss / n
    return total_loss


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
