# Paraphrase detection

## Usage

1. Install [conda](https://conda.io/docs/install/quick.html#id1) (I use miniconda on Linux)

2. Create the python environment

```
$ conda env create -f env.yml
```

3. Activate the environment
```
$ source activate ml
# On Windows: activate ml
```

4. Run

`jupyter notebook` to checkout the notebooks, `cd lstm && python train.py` to train the RNN. Might wanna change the variables at the top of `train.py`