# Pytorch Boilerplate

This is the repo I use as a boilerplate when I use pytorch.

## Features

- Automatically run on GPU or CPU
- Template for train/test/validate
- Calculate R2 accuracy
- Tensorboard logging
- Easy commandline flags
- Clear model definitions
- Clean data handling

## Install

I recommend using `conda` for python version and package management. The `conda` environment I use for most machine learning tasks is included in the repo. 

It uses python 3.6.1 and contains common ml packages (numpy, sklearn, pandas, pytorch, tensorflow, keras, etc). Conda can be installed from [here](https://conda.io/docs/install/quick.html) then the environment can be created with

```
$ conda env create -f environment.yml
```
and activated with
```
$ source activate ml  # linux / macOS
$ activate ml         # Windows
```

## Repo structure

The model(s) goes in `models.py`, data pre-processing / classes that inherit `torch.utils.data.Dataset` go in `data.py`, utility functions go in `utils.py`.

Model checkpoints will get dropped in `checkpoints/`, `tensorboard` logs go in `logs/runs/`. On each run a new folder is added with the name "%m-%d-%H:%M" (ex: `logs/runs/07-14-20:56`). Tensorboard logs are saved in format specific to tensorboard, models parameters are saved as `model_<epoch>.pt`, with the final saved model getting the name `model_last.pt`. `main.py` shows how to load a saved model.

Commandline arguments can be added in `args.py`

## Usage

There is an example in this repo that does linear regression on the Boston housing dataset using ADAM as an optimizer and MSE as a loss function (yes, it does poorly, just there so something can run).

(assumes env is already activated)

1. Start tensorboard

```
$ tensorboard --logdir=logs
```

2. Run the model
```
$ python main.py
```

3. Iterate on the model, optimizer, loss function, etc; run again
```
python main.py --n_epochs=10000 --lr=0.00001
```

Tensorboard is really good for comparing different models and approaches

![tensorboard example](static/tensorboard_ex.png)

## Typical workflow

My workflow for machine learning changes often, infact this repo is an attempt by me to more formalize my workflows so I can better improve them over time. That being said, feedback as to the structure of this repo, how I setup experiments, and so on is very welcome. This obviously isn't very detailed.

While iterating I also typically have `jupyter notebook`, I don't actually use the browser, I prefer the `jupyter` extension for VS Code. It allows you to connect to a `jupyter` kernel either local or remote and run code blocks from within a python file. If you see `# %%` in any files that I work on, that is the pattern to delimit the start of a jupyter cell. I find this allows me to run code quickly inline like I would in a browser notebook, but have the niceties of a modern text editor and keep all my code in python files instead of jumping between notebooks and files. Highly recommend the jupyter plugin for VS Code.

In order, what I usually touch is:

1. `data.py` to define new `Dataset`'s for the data I'm working with, since I find the pytorch `Dataloaders` the most convinient way to provide data to models. Then edit the Pre-process data section of main to reflect any changes.

2. `models.py` to build any models I'll be using, again, then I'll edit `main.py` with any changes that need to be made to properly instatiate the model, adding constants, etc at the top of `main.py`

3. Change optimizer and loss function in `main.py` as well as add any other metrics I'd like to track.