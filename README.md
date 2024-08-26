# LVNS-RAVE

This is the codebase for paper `LVNS-RAVE: Diversified audio generation with RAVE and Latent Vector Novelty Search` accepted as a poster to GECCO 2024.

ACM: https://dl.acm.org/doi/10.1145/3638530.3654432

arxiv: https://arxiv.org/abs/2404.14063


## Installation

```python
conda create -n eprior-rave python==3.9
conda activate eprior-rave
pip install -r requirements.txt
```

## Usage
### Prepare pretrained RAVEs
- [Official RAVE pretrained models](https://acids-ircam.github.io/rave_models_download)
- [IIL RAVEs](https://huggingface.co/Intelligent-Instruments-Lab/rave-models)

After download, put the `.ts` checkpoint file into `pretrained_models/`.

### Modify the LVNS config file

Find a template in `eprior/eprior/configs`. See `eprior/eprior/configs/test.gin` for comments of configs.

### Start evolution process

```python
cd /eprior/scripts

python run.py --config_path {your_config_file}

```
