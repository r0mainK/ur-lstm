# UR-LSTM

## Description

This repository revolves around the paper: [Improving the Gating Mechanism of Recurrent Neural Networks](https://arxiv.org/pdf/1910.09890.pdf) by Albert Gu, Caglar Gulcehre, Tom Paine, Matt Hoffman and Razvan Pascanu. 

In it, the authors introduce the **UR-LSTM**, a variant of the LSTM architecture which robustly improves the performance of the recurrent model, particularly when long-term dependencies are involved. 

Unfortunately, to my knowledge the authors did not release any code, either for the model or experiments - although they did provide pseudo-code for the model. Since I thought it was a really cool read, I decided to reimplement the model as well as some of the experiments with the Pytorch framework.

## Usage

If you simply want to use the model (the code is [here](src/models/ur_lstm.py)) then the only requirement is Pytorch. I haven't checked if it is compatible with older versions of the framework, but it _should_ be fine for everything past version `1.0`.  Following the Pytorch convention, it expects inputs with shape: `(seq_length, batch_size, embedding_dim)`. I did not implement the multi-layer variant but it should be pretty straightforward to do so if need be.

If you want to reproduce the experiments, you'll need to install a couple more things, so you should create a virtual environment and run from there, e.g.:

````bash
cd /path/to/the/repo
python3 -m venv .env # 3.6 or higher is advised
sourve .env/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
````

In order to configure a given experiment you can modify the related YAML file in the [here](src/configs/) directory. I've included the defaults for the UR-LSTM according to the paper (only doubt is for the embedding size of the language model) but you should play around for the vision and text experiments, as the configuration will not be optimal for the baselines. I've included information on each hyperparameter directly in the configurations.


## Experiments

I've coded both of the synthetic tasks, the image classification tasks on MNIST, permutated MNIST and CIFAR-10, and the language modeling task on WikiText103. At this point I don't intend on implementing the RL task or the additional experiments, but if someone wants to have a go feel free :)

Furthermore, I only included the ablated model (U-LSTM and R-LSTM) as well as the regular LSTM with modifiable forget bias. I'll probably come around to include the other baselines (chrono-init, cumax-activation, master gate) as well, but for now I haven't - again, PRs are welcome !

Except for the language model which prints test loss and perplexity before saving, I did not include any logging except on progress, via `tqdm`. You can however visualize the metrics directy in Tensorboard. All scripts will output the tensorboard logs in a `runs` directory at the base of the repo. To launch tensorboard, run this in a separate window:

````bash
cd /path/to/the/repo
sourve .env/bin/activate
tensorboard --logdir runs
````

Since everything is configured in the YAML files, to run a script simply do: `python3 script.py`. Models are not saved for the synthetic tasks, as they are pretty quick to train, and basically useless. By the way, if you do not have access to GPU compute you can still run the synthetic taks with lower sequence lengths (100-200) in reasonnable time, and see for yourslef the improvements brought by the architecture.

## License

[MIT](LICENSE)
