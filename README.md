# UR-LSTM

## Description

This repository revolves around the paper: [Improving the Gating Mechanism of Recurrent Neural Networks](https://arxiv.org/pdf/1910.09890.pdf) by Albert Gu, Caglar Gulcehre, Tom Paine, Matt Hoffman and Razvan Pascanu. 

In it, the authors introduce the **UR-LSTM**, a variant of the LSTM architecture which robustly improves the performance of the recurrent model, particularly when long-term dependencies are involved. 

Unfortunately, to my knowledge the authors did not release any code, either for the model or experiments - although they did provide pseudo-code for the model. Since I thought it was a really cool read, I decided to reimplement the model as well as some of the experiments with the Pytorch framework.

I've separated the code for the UR-LSTM, which is packaged and downloadable as a standalone module, from the code for the experiments. If you want to check out how to run them, go check [this page](experiments/README.md).

## Installation

With Python 3.6 or higher:

```bash
pip install ur-lstm-torch
```

I haven't checked if the model is compatible with older versions of Pytorch, but it _should_ be fine for everything past version `1.0`.

## Usage

The model can be used in the same way as the native `LSTM` implementation (doc is [here](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)), although I didn't implement the bidirectionnal variant and removed the `bias` keyword argument:

```python
import torch
from ur_lstm import URLSTM

input_size = 10
hidden_size = 20
num_layers = 2
batch_first = False
dropout = .5

model = URLSTM(
    input_size, hidden_size, num_layers=num_layers, batch_first=batch_first, dropout=dropout
)

batch_size = 3
seq_length = 5

x = torch.randn(seq_length, batch_size, input_size)
out, state = model(x)

print(out.shape) # (seq_length, batch_size, hidden_size)
print(len(state)) # 2, first is hidden state, second is cell state
print(state[0].shape) # (num_layers, batch_size, hidden_size)
print(state[1].shape) # (num_layers, batch_size, hidden_size)
```

If you want to implement a custom model, you can also import and use the `URLSTMCell` module in the same way you would the regular `LSTMCell` (doc is [here](https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html#torch.nn.LSTMCell)), although again I removed the `bias` keyword argument:


```python
import torch
from ur_lstm import URLSTMCell

input_size = 10
hidden_size = 20

cell = URLSTMCell(input_size, hidden_size)

batch_size = 2

x = torch.randn(batch_size, input_size)
state = torch.randn(batch_size, hidden_size), torch.randn(batch_size, hidden_size)
out, state = cell(x, state)

print(out.shape) # (batch_size, hidden_size)
print(len(state)) # 2, first is hidden state, second is cell state
print(state[0].shape) # (batch_size, hidden_size)
print(state[1].shape) # (num_layers, batch_size, hidden_size)
```

## License

[MIT](LICENSE)
