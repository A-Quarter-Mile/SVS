# Muskits SVS Recipe TEMPLATE

This is a template of SVS recipe for Muskits.

## Table of Contents

* [Muskits SVS Recipe TEMPLATE](#muskits-svs-recipe-template)
  * [Table of Contents](#table-of-contents)
  * [Recipe flow](#recipe-flow)
    * [1\. Data preparation](#1-data-preparation)
    * [2\. Wav dump / Feature extract & Embedding preparation](#2-wav-dump--feature-extract--embedding-preparation)
    * [3\. Removal of long / short data](#3-removal-of-long--short-data)
    * [4\. Token list generation](#4-token-list-generation)
    * [5\. SVS statistics collection](#5-svs-statistics-collection)
    * [6\. SVS training](#6-svs-training)
    * [7\. SVS decoding](#7-svs-decoding)
    * [8\. Pack results](#8-pack-results)
  * [How to run](#how-to-run)
    * [Multi speaker model with speaker ID embedding training](#multi-speaker-model-with-speaker-id-embedding-training)
    * [Multi language model with language ID embedding training](#multi-language-model-with-language-id-embedding-training)
    * [Vocoder training](#vocoder-training)
    * [Evaluation](#evaluation)
  * [Supported text frontend](#supported-text-frontend)
  * [Supported text cleaner](#supported-text-cleaner)
  * [Supported Models](#supported-models)


## Recipe flow

SVS recipe consists of 9 stages.

### 1. Data preparation

Data preparation stage.
It calls `local/data.sh` to creates Kaldi-style data directories in `data/` for training, validation, and evaluation sets.

Each directory of training set, development set, and evaluation set, has same directory structure. See also http://kaldi-asr.org/doc/data_prep.html about Kaldi data structure. 

### 2. Wav dump / Feature extract & Embedding preparation

If you specify `--feats_type raw` option, this is a wav dumping stage which reformats `wav.scp` in data directories.
Else, if you specify `--feats_type fbank` option or `--feats_type stft` option, this is a feature extracting stage (to be updated).

Also, speaker ID embedding and language ID embedding preparation will be performed in this stage if you specify `--use_sid true` and `--use_lid true` options.
Note that this processing assume that `utt2spk` or `utt2lang` are correctly created in stage 1, please be careful.

### 3. Removal of long / short data

Processing stage to remove long and short utterances from the training and validation data. 
You can change the threshold values via `--min_wav_duration` and `--max_wav_duration`.

Empty text will also be removed.

### 4. Token list generation

Token list generation stage.
It generates token list (dictionary) from `srctexts`.
You can change the tokenization type via `--token_type` option.
`token_type=char` and `token_type=phn` are supported.
If `--cleaner` option is specified, the input text will be cleaned with the specified cleaner.
If `token_type=phn`, the input text will be converted with G2P module specified by `--g2p` option.

See also:
- [Supported text cleaner](#supported-text-cleaner).
- [Supported text frontend](#supported-text-frontend).

Data preparation will end in stage 4. You can skip data preparation (stage 1 ~ stage 4) via `--skip_data_prep` option.

### 5. SVS statistics collection

Statistics calculation stage.
It collects the shape information of the input and output and calculates statistics for feature normalization (mean and variance over training data).

### 6. SVS training

SVS model training stage.
You can change the training setting via `--train_config` and `--train_args` options.

See also:
- [Supported models](#supported-models).
- [Change the configuration for training](https://espnet.github.io/espnet/espnet2_training_option.html)

Training process will end in stage 6. You can skip training process (stage 5 ~ stage 6) via `--skip_train` option.

### 7. SVS decoding

SVS model decoding stage.
You can change the decoding setting via `--inference_config` and `--inference_args`.

See also:
- [Change the configuration for training](https://espnet.github.io/espnet/espnet2_training_option.html)

### 8. Pack results

Packing stage.
It packs the trained model files.


## How to run

See [Tutorial](https://github.com/SJTMusicTeam/Muskits/blob/main/doc/tutorial.md#muskits).

As a default, we train ofuton_p_utagoe (`conf/train.yaml`) with `feats_type=raw` + `token_type=phn`.

Then, you can get the following directories in the recipe directory.
```sh
├── data/ # Kaldi-style data directory
│   ├── dev/           # validation set
│   ├── eval/          # evaluation set
│   ├── tr_no_dev/     # training set
│   └── token_list/    # token list (directory)
│        └── phn_none_jp/  # token list
├── dump/ # feature dump directory
│   └── raw/
│       ├── org/
│       │    ├── tr_no_dev/ # training set before filtering
│       │    └── dev/       # validation set before filtering
│       ├── srctexts   # text to create token list
│       ├── eval/      # evaluation set
│       ├── dev/       # validation set after filtering
│       └── tr_no_dev/ # training set after filtering
└── exp/ # experiment directory
    ├── svs_stats_raw_phn_none_jp  # statistics
    └── svs_train_raw_phn_none_jp  # model
        ├── tensorboard/           # tensorboard log
        ├── images/                # plot of training curves
        ├── valid/                 # valid results
        ├── decode_train.loss.best/ # decoded results
        │    ├── dev/   # validation set
        │    └── eval/ # evaluation set
        │        ├── norm/        # generated features
        │        ├── denorm/      # generated denormalized features
        │        ├── wav/         # generated wav via Griffin-Lim
        │        ├── log/         # log directory
        │        ├── feats_type   # feature type
        │        └── speech_shape # shape info of generated features
        ├── config.yaml             # config used for the training
        ├── train.log               # training log
        ├── *epoch.pth              # model parameter file
        ├── checkpoint.pth          # model + optimizer + scheduler parameter file
        ├── latest.pth              # symlink to latest model parameter
        ├── *.ave_2best.pth         # model averaged parameters
        └── *.best.pth              # symlink to the best model parameter loss
```

For the first time, we recommend performing each stage step-by-step via `--stage` and `--stop-stage` options.
```sh
$ ./run.sh --stage 1 --stop-stage 1
$ ./run.sh --stage 2 --stop-stage 2
...
$ ./run.sh --stage 7 --stop-stage 7
```
This might helps you to understand each stage's processing and directory structure.


### Multi-speaker model with speaker ID embedding training

First, you need to run from the stage 2 and 3 with `--use_sid true` to extract speaker ID.
```sh
$ ./run.sh --stage 2 --stop-stage 3 --use_sid true
```
You can find the speaker ID file in `dump/raw/*/utt2sid`.
Note that you need to correctly create `utt2spk` in data prep stage to generate `utt2sid`.
Then, you can run the training with the config which has `spks: #spks` in `svs_conf`.
```yaml
# e.g.
svs_conf:
    spks: 5  # Number of speakers
```
Please run the training from stage 6.
```sh
$ ./run.sh --stage 6 --use_sid true --train_config /path/to/your_multi_spk_config.yaml
```

### Multi-language model with language ID embedding training

First, you need to run from the stage 2 and 3 with `--use_lid true` to extract speaker ID.
```sh
$ ./run.sh --stage 2 --stop-stage 3 --use_lid true
```
You can find the speaker ID file in `dump/raw/*/utt2lid`.
**Note that you need to additionally create `utt2lang` file in data prep stage to generate `utt2lid`.**
Then, you can run the training with the config which has `langs: #langs` in `svs_conf`.
```yaml
# e.g.
svs_conf:
    langs: 4  # Number of languages
```
Please run the training from stage 6.
```sh
$ ./run.sh --stage 6 --use_lid true --train_config /path/to/your_multi_lang_config.yaml
```

Of course you can further combine with speaker ID embedding.
If you want to use both sid and lid, the process should be like this:
```sh
$ ./run.sh --stage 2 --stop-stage 3 --use_lid true --use_sid true
```
Make your config.
```yaml
# e.g.
svs_conf:
    langs: 4   # Number of languages
    spks: 5    # Number of speakers
```
Please run the training from stage 6.
```sh
$ ./run.sh --stage 6 --use_lid true --use_sid true --train_config /path/to/your_multi_spk_multi_lang_config.yaml
```


### Vocoder training

If you `--vocoder_file` is set to none, Griffin-Lim will be used.
You can also train corresponding vocoder using [kan-bayashi/ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)..

Pretrained vocoder is like follows:

```sh
*_hifigan.v1 
├── checkpoint-xxxxxxsteps.pkl
├── config.yml
└── stats.h5
```

```sh
# Use the vocoder trained by `parallel_wavegan` repo manually
$ ./run.sh --stage 7 --vocoder_file /path/to/checkpoint-xxxxxxsteps.pkl --inference_tag decode_with_my_vocoder
```

### Evaluation

We provide a objective evaluation metrics:

- Mel-cepstral distortion (MCD)

For MCD, we apply dynamic time-warping (DTW) to match the length difference between ground-truth speech and generated speech.

Here we show the example command to calculate objective metrics:

```sh
cd egs/<recipe_name>/svs1
. ./path.sh
# Evaluate MCD
./pyscripts/utils/evaluate_mcd.py \
    exp/<model_dir_name>/<decode_dir_name>/eval/wav/wav.scp \
    dump/raw/eval/wav.scp
```
While these objective metrics can estimate the quality of synthesized speech, it is still difficult to fully determine human perceptual quality from these values, especially with high-fidelity generated speech.
Therefore, we recommend performing the subjective evaluation if you want to check perceptual quality in detail.

You can refer [this page](https://github.com/kan-bayashi/webMUSHRA/blob/master/HOW_TO_SETUP.md) to launch web-based subjective evaluation system with [webMUSHRA](https://github.com/audiolabs/webMUSHRA).


## Supported text frontend

You can change via `--g2p` option in `svs.sh`.

- `none`: Just separate by space
    - e.g.: `HH AH0 L OW1 <space> W ER1 L D` -> `[HH, AH0, L, OW1, <space>, W, ER1, L D]`

You can see the code example from [here](https://github.com/SJTMusicTeam/Muskits/blob/main/muskit/text/phoneme_tokenizer.py).


## Supported text cleaner

You can change via `--cleaner` option in `svs.sh`.

- `none`: No text cleaner.

You can see the code example from [here](https://github.com/SJTMusicTeam/Muskits/blob/main/muskit/text/cleaner.py).

## Supported Models

You can train the following models by changing `*.yaml` config for `--train_config` option in `run.sh`.

- [Naive-RNN]()
- [GLU-Transformer]()
- [MLP-Singer](https://arxiv.org/abs/2106.07886)
- [XiaoIce](https://arxiv.org/pdf/2006.06261)

You can find example configs of the above models in [`egs/ofuton_p_utagoe_db/svs1/conf/tuning`](../../ofuton_p_utagoe_db/svs1/conf/tuning).



