
# Experiments

## Setup

To reproduce the experiments, you'll need to install a couple things, so you should create a virtual environment and run from there, e.g.:

````bash
cd /path/to/the/repo/experiments
python3 -m venv .env # 3.6 or higher is advised
source .env/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
````

## Configuration

In order to configure a given experiment you can modify the related YAML file in [this directory](configs/). I've included the defaults for the UR-LSTM according to the paper (my only doubt is for the embedding size of the language model) but you should play around for the vision and text experiments, as the configuration will not be optimal for the baselines. I've included information on each hyperparameter directly in the configuration files, but feel free to open an issue if something is unclear.

## Usage

I've coded both of the synthetic tasks, the image classification tasks on MNIST, permutated MNIST and CIFAR-10, and the language modeling task on WikiText103. At this point I don't intend on implementing the RL task or the additional experiments, but if someone wants to have a go feel free :)

Regarding baselines, I only included the ablated model (U-LSTM and R-LSTM) as well as the regular LSTM with modifiable forget bias. I'll probably come around to include the other baselines (chrono-init, cumax-activation, master gate) as well, but for now I haven't - again, PRs are welcome !

Except for the language model which prints the test loss and perplexity before saving the final model, I did not include any logging except on progress, via `tqdm`. You can however visualize the metrics directy in Tensorboard. All scripts will output the Tensorboard logs in a `runs` directory at the base of the `experiments` folder, so to launch Tensorboard do something like this in a separate window:

````bash
cd /path/to/the/repo/experiments
sourve .env/bin/activate
tensorboard --logdir runs
````

Since everything is configured in the YAML files, to run a script you simply do: `python3 script.py`. Models are not saved for the synthetic tasks, as they are pretty quick to train and basically useless. For the other tasks the model `state_dict` is saved, and for the language model the vocabulary is also saved as a list ordered by token index.

If you do not have access to GPU compute running the experiments will be either impossible or quite long. However, you can still run the synthetic tasks with lower sequence lengths (100~200) in reasonnable time, and see for yourself the improvements brought by the architecture.
