# Muskits SVS Recipe TEMPLATE

This is a template of SVS recipe for Muskits.

## Table of Contents

* [Muskits SVS Recipe TEMPLATE](#muskits-svs-recipe-template)
  * [Table of Contents](#table-of-contents)
  * [Recipe flow](#recipe-flow)
    * [1\. Data preparation](#1-data-preparation)
    * [2\. Wav dump / Embedding preparation](#2-wav-dump--embedding-preparation)
    * [3\. Removal of long / short data](#3-removal-of-long--short-data)
    * [4\. Token list generation](#4-token-list-generation)
    * [5\. SVS statistics collection](#5-svs-statistics-collection)
    * [6\. SVS training](#6-svs-training)
    * [7\. SVS decoding](#7-svs-decoding)
    * [8\. (Optional) Pack results for upload](#8-optional-pack-results-for-upload)
  * [How to run](#how-to-run)
    * [FastSpeech training](#fastspeech-training)
    * [FastSpeech2 training](#fastspeech2-training)
    * [Multi speaker model with X-vector training](#multi-speaker-model-with-x-vector-training)
    * [Multi speaker model with speaker ID embedding training](#multi-speaker-model-with-speaker-id-embedding-training)
    * [Multi language model with language ID embedding training](#multi-language-model-with-language-id-embedding-training)
    * [VITS training](#vits-training)
    * [Joint text2wav training](#joint-text2wav-training)
    * [Evaluation](#evaluation)
  * [Supported text frontend](#supported-text-frontend)
  * [Supported text cleaner](#supported-text-cleaner)
  * [Supported Models](#supported-models)
    * [Single speaker model](#single-speaker-model)
    * [Multi speaker model](#multi-speaker-model)
  * [FAQ](#faq)
    * [ESPnet1 model is compatible with ESPnet2?](#espnet1-model-is-compatible-with-espnet2)
    * [How to change minibatch size in training?](#how-to-change-minibatch-size-in-training)
    * [How to make a new recipe for my own dataset?](#how-to-make-a-new-recipe-for-my-own-dataset)
    * [How to add a new g2p module?](#how-to-add-a-new-g2p-module)
    * [How to add a new cleaner module?](#how-to-add-a-new-cleaner-module)
    * [How to use trained model in python?](#how-to-use-trained-model-in-python)
    * [How to get pretrained models?](#how-to-get-pretrained-models)
    * [How to load the pretrained parameters?](#how-to-load-the-pretrained-parameters)
    * [How to finetune the pretrained model?](#how-to-finetune-the-pretrained-model)
    * [How to add a new model?](#how-to-add-a-new-model)
    * [How to test my model with an arbitrary given text?](#how-to-test-my-model-with-an-arbitrary-given-text)
    * [How to train vocoder?](#how-to-train-vocoder)
    * [How to train vocoder with text2mel GTA outputs?](#how-to-train-vocoder-with-text2mel-gta-outputs)
    * [How to handle the errors in validate_data_dir.sh?](#how-to-handle-the-errors-in-validate_data_dirsh)
    * [Why the model generate meaningless speech at the end?](#why-the-model-generate-meaningless-speech-at-the-end)
    * [Why the model cannot be trained well with my own dataset?](#why-the-model-cannot-be-trained-well-with-my-own-dataset)
    * [Why the outputs contains metallic noise when combining neural vocoder?](#why-the-outputs-contains-metallic-noise-when-combining-neural-vocoder)
    * [How is the duration for FastSpeech2 generated?](#how-is-the-duration-for-fastspeech2-generated)
    * [Why the output of Tacotron2 or Transformer is non-deterministic?](#why-the-output-of-tacotron2-or-transformer-is-non-deterministic)

## Recipe flow

TTS recipe consists of 9 stages.

### 1. Data preparation

Data preparation stage.
It calls `local/data.sh` to creates Kaldi-style data directories in `data/` for training, validation, and evaluation sets.

See also:
- [About Kaldi-style data directory](https://github.com/espnet/espnet/tree/master/egs2/TEMPLATE#about-kaldi-style-data-directory)

### 2. Wav dump / Embedding preparation

Wav dumping stage.
This stage reformats `wav.scp` in data directories.

Additionally, We support X-vector extraction in this stage as you can use in ESPnet1.
If you specify `--use_xvector true` (Default: `use_xvector=false`), we extract X-vectors.
You can select the type of toolkit to use (kaldi, speechbrain, or espnet) when you specify `--xvector_tool <option>` 
(Default: `xvector_tool=kaldi`).
If you specify kaldi, then we additionally extract mfcc features and vad decision.
This processing requires the compiled kaldi, please be careful.

Also, speaker ID embedding and language ID embedding preparation will be performed in this stage if you specify `--use_sid true` and `--use_lid true` options.
Note that this processing assume that `utt2spk` or `utt2lang` are correctly created in stage 1, please be careful.

### 3. Removal of long / short data

Processing stage to remove long and short utterances from the training and validation data.
You can change the threshold values via `--min_wav_duration` and `--max_wav_duration`.

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

### 5. SVS statistics collection

Statistics calculation stage.
It collects the shape information of the input and output and calculates statistics for feature normalization (mean and variance over training data).

### 6. SVS training

TTS model training stage.
You can change the training setting via `--train_config` and `--train_args` options.

See also:
- [Supported models](#supported-models).
- [Change the configuration for training](https://espnet.github.io/espnet/espnet2_training_option.html)
- [Distributed training](https://espnet.github.io/espnet/espnet2_distributed.html)

### 7. SVS decoding

TTS model decoding stage.
You can change the decoding setting via `--inference_config` and `--inference_args`.

See also:
- [Change the configuration for training](https://espnet.github.io/espnet/espnet2_training_option.html)

### 8. (Optional) Pack results for upload

Packing stage.
It packs the trained model files and uploads to [Zenodo](https://zenodo.org/) (Zenodo upload will be deprecated).
If you want to run this stage, you need to register your account in zenodo.

See also:
- [ESPnet Model Zoo](https://github.com/espnet/espnet_model_zoo)

#### Stage 10: Upload model to Hugging Face

Upload the trained model to Hugging Face for sharing. Additional information at [Docs](https://espnet.github.io/espnet/espnet2_tutorial.html#packing-and-sharing-your-trained-model).

## How to run

Here, we show the procedure to run the recipe using `egs2/ljspeech/tts1`.

Move on the recipe directory.
```sh
$ cd egs2/ljspeech/tts1
```

Modify `LJSPEECH` variable in `db.sh` if you want to change the download directory.
```sh
$ vim db.sh
```

Modify `cmd.sh` and `conf/*.conf` if you want to use job scheduler.
See the detail in [using job scheduling system](https://espnet.github.io/espnet/parallelization.html).
```sh
$ vim cmd.sh
```

Run `run.sh`, which conducts all of the stages explained above.
```sh
$ ./run.sh
```
As a default, we train Tacotron2 (`conf/train.yaml`) with `feats_type=raw` + `token_type=phn`.

Then, you can get the following directories in the recipe directory.
```sh
├── data/ # Kaldi-style data directory
│   ├── dev/        # validation set
│   ├── eval1/      # evaluation set
│   └── tr_no_dev/  # training set
├── dump/ # feature dump directory
│   ├── token_list/    # token list (dictionary)
│   └── raw/
│       ├── org/
│       │    ├── tr_no_dev/ # training set before filtering
│       │    └── dev/       # validation set before filtering
│       ├── srctexts   # text to create token list
│       ├── eval1/     # evaluation set
│       ├── dev/       # validation set after filtering
│       └── tr_no_dev/ # training set after filtering
└── exp/ # experiment directory
    ├── tts_stats_raw_phn_tacotron_g2p_en_no_space # statistics
    └── tts_train_raw_phn_tacotron_g2p_en_no_space # model
        ├── att_ws/                # attention plot during training
        ├── tensorboard/           # tensorboard log
        ├── images/                # plot of training curves
        ├── decode_train.loss.ave/ # decoded results
        │    ├── dev/   # validation set
        │    └── eval1/ # evaluation set
        │        ├── att_ws/      # attention plot in decoding
        │        ├── probs/       # stop probability plot in decoding
        │        ├── norm/        # generated features
        │        ├── denorm/      # generated denormalized features
        │        ├── wav/         # generated wav via Griffin-Lim
        │        ├── log/         # log directory
        │        ├── durations    # duration of each input tokens
        │        ├── feats_type   # feature type
        │        ├── focus_rates  # focus rate
        │        └── speech_shape # shape info of generated features
        ├── config.yaml             # config used for the training
        ├── train.log               # training log
        ├── *epoch.pth              # model parameter file
        ├── checkpoint.pth          # model + optimizer + scheduler parameter file
        ├── latest.pth              # symlink to latest model parameter
        ├── *.ave_5best.pth         # model averaged parameters
        └── *.best.pth              # symlink to the best model parameter loss
```
In decoding, we use Griffin-Lim for waveform generation as a default (End-to-end text-to-wav model can generate waveform directly such as VITS and Joint training model).
If you want to combine with neural vocoders, you can combine with [kan-bayashi/ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN).

```sh
# Make sure you already install parallel_wavegan repo
$ . ./path.sh && pip install -U parallel_wavegan
# Use parallel_wavegan provided pretrained ljspeech style melgan as a vocoder
$ ./run.sh --stage 7 --inference_args "--vocoder_tag parallel_wavegan/ljspeech_style_melgan.v1" --inference_tag decode_with_ljspeech_style_melgan.v1
# Use the vocoder trained by `parallel_wavegan` repo manually
$ ./run.sh --stage 7 --vocoder_file /path/to/checkpoint-xxxxxxsteps.pkl --inference_tag decode_with_my_vocoder
```

If you want to generate waveform from dumped features, please check [decoding with ESPnet-TTS model's feature](https://github.com/kan-bayashi/ParallelWaveGAN#decoding-with-espnet-tts-models-features).

For the first time, we recommend performing each stage step-by-step via `--stage` and `--stop-stage` options.
```sh
$ ./run.sh --stage 1 --stop-stage 1
$ ./run.sh --stage 2 --stop-stage 2
...
$ ./run.sh --stage 7 --stop-stage 7
```
This might helps you to understand each stage's processing and directory structure.

### FastSpeech training

If you want to train FastSpeech, additional steps with the teacher model are needed.
Please make sure you already finished the training of the teacher model (Tacotron2 or Transformer-TTS).

First, decode all of data including training, validation, and evaluation set.
```sh
# specify teacher model directory via --tts_exp option
$ ./run.sh --stage 7 \
    --tts_exp exp/tts_train_raw_phn_tacotron_g2p_en_no_space \
    --test_sets "tr_no_dev dev eval1"
```
This will generate `durations` for training, validation, and evaluation sets in `exp/tts_train_raw_phn_tacotron_g2p_en_no_space/decode_train.loss.ave`.

Then, you can train FastSpeech by specifying the directory including `durations` via `--teacher_dumpdir` option.
```sh
$ ./run.sh --stage 6 \
    --train_config conf/tuning/train_fastspeech.yaml \
    --teacher_dumpdir exp/tts_train_raw_phn_tacotron_g2p_en_no_space/decode_train.loss.ave
```

In the above example, we use generated mel-spectrogram as the target, which is known as knowledge distillation training.
If you want to use groundtruth mel-spectrogram as the target, we need to use teacher forcing in decoding.
```sh
$ ./run.sh --stage 7 \
    --tts_exp exp/tts_train_raw_phn_tacotron_g2p_en_no_space \
    --inference_args "--use_teacher_forcing true" \
    --test_sets "tr_no_dev dev eval1"
```
You can get the groundtruth aligned durations in `exp/tts_train_raw_phn_tacotron_g2p_en_no_space/decode_use_teacher_forcingtrue_train.loss.ave`.

Then, you can train FastSpeech without knowledge distillation.
```sh
$ ./run.sh --stage 6 \
    --train_config conf/tuning/train_fastspeech.yaml \
    --teacher_dumpdir exp/tts_train_raw_phn_tacotron_g2p_en_no_space/decode_use_teacher_forcingtrue_train.loss.ave
```

### FastSpeech2 training

The procedure is almost the same as FastSpeech but we **MUST** use teacher forcing in decoding.
```sh
$ ./run.sh --stage 7 \
    --tts_exp exp/tts_train_raw_phn_tacotron_g2p_en_no_space \
    --inference_args "--use_teacher_forcing true" \
    --test_sets "tr_no_dev dev eval1"
```

To train FastSpeech2, we use additional feature (F0 and energy).
Therefore, we need to start from `stage 5` to calculate additional statistics.
```sh
$ ./run.sh --stage 5 \
    --train_config conf/tuning/train_fastspeech2.yaml \
    --teacher_dumpdir exp/tts_train_raw_phn_tacotron_g2p_en_no_space/decode_use_teacher_forcingtrue_train.loss.ave \
    --tts_stats_dir exp/tts_train_raw_phn_tacotron_g2p_en_no_space/decode_use_teacher_forcingtrue_train.loss.ave/stats \
    --write_collected_feats true
```
where `--tts_stats_dir` is the option to specify the directory to dump Statistics, and `--write_collected_feats` is the option to dump features in statistics calculation.
The use of `--write_collected_feats` is optional but it helps to accelerate the training.

### Multi-speaker model with X-vector training

First, you need to run from the stage 2 and 3 with `--use_xvector true` to extract X-vector.
```sh
$ ./run.sh --stage 2 --stop-stage 3 --use_xvector true
```
You can find the extracted X-vector in `dump/xvector/*/xvector.{ark,scp}`.
Then, you can run the training with the config which has `spk_embed_dim: 512` in `tts_conf`.
```yaml
# e.g.
tts_conf:
    spk_embed_dim: 512               # dimension of speaker embedding
    spk_embed_integration_type: add  # how to integrate speaker embedding
```
Please run the training from stage 6.
```sh
$ ./run.sh --stage 6 --use_xvector true --train_config /path/to/your_xvector_config.yaml
```

You can find the example config in [`egs2/vctk/tts1/conf/tuning`](../../vctk/tts1/conf/tuning).

### Multi-speaker model with speaker ID embedding training

First, you need to run from the stage 2 and 3 with `--use_sid true` to extract speaker ID.
```sh
$ ./run.sh --stage 2 --stop-stage 3 --use_sid true
```
You can find the speaker ID file in `dump/raw/*/utt2sid`.
Note that you need to correctly create `utt2spk` in data prep stage to generate `utt2sid`.
Then, you can run the training with the config which has `spks: #spks` in `tts_conf`.
```yaml
# e.g.
tts_conf:
    spks: 128  # Number of speakers
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
Then, you can run the training with the config which has `langs: #langs` in `tts_conf`.
```yaml
# e.g.
tts_conf:
    langs: 4  # Number of languages
```
Please run the training from stage 6.
```sh
$ ./run.sh --stage 6 --use_lid true --train_config /path/to/your_multi_lang_config.yaml
```

Of course you can further combine with x-vector or speaker ID embedding.
If you want to use both sid and lid, the process should be like this:
```sh
$ ./run.sh --stage 2 --stop-stage 3 --use_lid true --use_sid true
```
Make your config.
```yaml
# e.g.
tts_conf:
    langs: 4   # Number of languages
    spks: 128  # Number of speakers
```
Please run the training from stage 6.
```sh
$ ./run.sh --stage 6 --use_lid true --use_sid true --train_config /path/to/your_multi_spk_multi_lang_config.yaml
```

### VITS training

First, the VITS config is **hard coded for 22.05 khz or 44.1 khz** and use different feature extraction method.
(Note that you can use any feature extraction method but the default method is `linear_spectrogram`.)
If you want to use it with 24 khz or 16 khz dataset, please be careful about these point.

```sh
# Assume that data prep stage (stage 1) is finished
$ ./run.sh --stage 1 --stop-stage 1
# Single speaker 22.05 khz case
$ ./run.sh \
    --stage 2 \
    --ngpu 4 \
    --fs 22050 \
    --n_fft 1024 \
    --n_shift 256 \
    --win_length null \
    --dumpdir dump/22k \
    --expdir exp/22k \
    --tts_task gan_tts \
    --feats_extract linear_spectrogram \
    --feats_normalize none \
    --train_config ./conf/tuning/train_vits.yaml \
    --inference_config ./conf/tuning/decode_vits.yaml \
    --inference_model latest.pth
# Single speaker 44.1 khz case
$ ./run.sh \
    --stage 2 \
    --ngpu 4 \
    --fs 44100 \
    --n_fft 2048 \
    --n_shift 512 \
    --win_length null \
    --dumpdir dump/44k \
    --expdir exp/44k \
    --tts_task gan_tts \
    --feats_extract linear_spectrogram \
    --feats_normalize none \
    --train_config ./conf/tuning/train_full_band_vits.yaml \
    --inference_config ./conf/tuning/decode_vits.yaml \
    --inference_model latest.pth
# Multi speaker with SID 22.05 khz case
$ ./run.sh \
    --stage 2 \
    --use_sid true \
    --ngpu 4 \
    --fs 22050 \
    --n_fft 1024 \
    --n_shift 256 \
    --win_length null \
    --dumpdir dump/22k \
    --expdir exp/22k \
    --tts_task gan_tts \
    --feats_extract linear_spectrogram \
    --feats_normalize none \
    --train_config ./conf/tuning/train_multi_spk_vits.yaml \
    --inference_config ./conf/tuning/decode_vits.yaml \
    --inference_model latest.pth
# Multi speaker with SID 44.1 khz case
$ ./run.sh \
    --stage 2 \
    --use_sid true \
    --ngpu 4 \
    --fs 44100 \
    --n_fft 2048 \
    --n_shift 512 \
    --win_length null \
    --dumpdir dump/44k \
    --expdir exp/44k \
    --tts_task gan_tts \
    --feats_extract linear_spectrogram \
    --feats_normalize none \
    --train_config ./conf/tuning/train_full_band_multi_spk_vits.yaml \
    --inference_config ./conf/tuning/decode_vits.yaml \
    --inference_model latest.pth
# Multi speaker with X-vector 22.05 khz case (need compiled kaldi to run)
$ ./run.sh \
    --stage 2 \
    --use_xvector true \
    --ngpu 4 \
    --fs 22050 \
    --n_fft 1024 \
    --n_shift 256 \
    --win_length null \
    --dumpdir dump/22k \
    --expdir exp/22k \
    --tts_task gan_tts \
    --feats_extract linear_spectrogram \
    --feats_normalize none \
    --train_config ./conf/tuning/train_xvector_vits.yaml \
    --inference_config ./conf/tuning/decode_vits.yaml \
    --inference_model latest.pth
```

The training time requires long times (around several weeks) but around 100k samples can generate a reasonable sounds.

You can find the example configs in:
- [`egs2/ljspeech/tts1/conf/tuning/train_vits.yaml`: Single speaker 22.05 khz config](../../ljspeech/tts1/conf/tuning/train_vits.yaml).
- [`egs2/jsut/tts1/conf/tuning/train_full_band_vits.yaml`: Single speaker 44.1 khz config](../../jsut/tts1/conf/tuning/train_full_band_vits.yaml).
- [`egs2/vctk/tts1/conf/tuning/train_multi_spk_vits.yaml`: Multi speaker with SID 22.05 khz config](../../vctk/tts1/conf/tuning/train_multi_spk_vits.yaml).
- [`egs2/vctk/tts1/conf/tuning/train_full_band_multi_spk_vits.yaml`: Multi speaker with SID 44.1 khz config](../../vctk/tts1/conf/tuning/train_full_band_multi_spk_vits.yaml).
- [`egs2/libritts/tts1/conf/tuning/train_xvector_vits.yaml`: Multi speaker with X-vector 22.05 khz config](../../libritts/tts1/conf/tuning/train_xvector_vits.yaml).

### Joint text2wav training

Joint training enables us to train both text2mel and vocoder model jointly with GAN-based training.
Currently, we tested on only for non-autoregressive text2mel models with ljspeech dataset but the following models and vocoders are supported.

**Text2mel**

- Tacotron2
- Transformer
- FastSpeech
- FastSpeech2

**Vocoder**

- ParallelWaveGAN G / D
- (Multi-band) MelGAN G / D
- HiFiGAN G / D
- StyleMelGAN G / D

Here, we show the example procedure to train conformer fastspeech2 + hifigan jointly with two training strategy (training from scratch and fine-tuning of pretrained text2mel and vocoder).

```sh
# Make sure you are ready to train fastspeech2 (already prepared durations file with teacher model)
$ ...
# Case 1: Train conformer fastspeech2 + hifigan G + hifigan D from scratch
$ ./run.sh \
    --stage 6 \
    --tts_task gan_tts \
    --train_config ./conf/tuning/train_joint_conformer_fastspeech2_hifigan.yaml
# Case 2: Fine-tuning of pretrained conformer fastspeech2 + hifigan G + hifigan D
# (a) Prepare pretrained models as follows
$ tree -L 2 exp
exp
...
├── ljspeech_hifigan.v1  # pretrained vocoder
│   ├── checkpoint-2500000steps.pkl
│   ├── config.yml
│   └── stats.h5
├── tts_train_conformer_fastspeech2_raw_phn_tacotron_g2p_en_no_space  # pretrained text2mel
│   ├── config.yaml
│   ├── images
│   └── train.loss.ave_5best.pth
...
# If you want to use the same files of this example
$ ipython
# Download text2mel model
[ins] In [1]: from espnet_model_zoo.downloader import ModelDownloader
[ins] In [2]: d = ModelDownloader("./downloads")
[ins] In [3]: d.download_and_unpack("kan-bayashi/ljspeech_conformer_fastspeech2")
# Download vocoder
[ins] In [4]: from parallel_wavegan.utils import download_pretrained_model
[ins] In [5]: download_pretrained_model("ljspeech_hifigan.v1", "downloads")
# Move them to exp directory
$ mv download/59c43ac0d40b121060bd71dd418f5ece/exp/tts_train_conformer_fastspeech2_raw_phn_tacotron_g2p_en_no_space exp
$ mv downloads/ljspeech_hifigan.v1 exp
# (b) Convert .pkl checkpoint to espnet loadable format
$ ipython
[ins] In [1]: import torch
[ins] In [2]: d = torch.load("./exp/ljspeech_hifigan.v1/checkpoint-2500000steps.pkl")
[ins] In [3]: torch.save(d["model"]["generator"], "generator.pth")
[ins] In [4]: torch.save(d["model"]["discriminator"], "discriminator.pth")
# (c) Prepare configuration
$ vim conf/tuning/finetune_joint_conformer_fastspeech2_hifigan.yaml
# edit text2mel_params / generator_params / discriminator_params to be the same as the pretrained model
# edit init_param part to specify the correct path of the pretrained model
# (d) Run training
$ ./run.sh \
    --stage 6 \
    --tts_task gan_tts \
    --train_config ./conf/tuning/finetune_joint_conformer_fastspeech2_hifigan.yaml
```

You can find the example configs in:
- [`egs2/ljspeech/tts1/conf/tuning/train_joint_conformer_fastspeech2_hifigan.yaml`: Joint training of conformer fastspeech2 + hifigan](../../ljspeech/tts1/conf/tuning/train_joint_conformer_fastspeech2_hifigan.yaml).
- [`egs2/ljspeech/tts1/conf/tuning/finetune_joint_conformer_fastspeech2_hifigan.yaml`: Joint fine-tuning of conformer fastspeech2 + hifigan](../../ljspeech/tts1/conf/tuning/finetune_joint_conformer_fastspeech2_hifigan.yaml).

### Evaluation

We provide three objective evaluation metrics:

- Mel-cepstral distortion (MCD)
- Log-F0 root mean square error (log-F0 RMSE)
- Character error rate (CER)

MCD and log-F0 RMSE reflect speaker, prosody, and phonetic content similarities, and CER can reflect the intelligibility.
For MCD and log-F0 RMSE, we apply dynamic time-warping (DTW) to match the length difference between ground-truth speech and generated speech.

Here we show the example command to calculate objective metrics:

```sh
cd egs2/<recipe_name>/tts1
. ./path.sh
# Evaluate MCD
./pyscripts/utils/evaluate_mcd.py \
    exp/<model_dir_name>/<decode_dir_name>/eval1/wav/wav.scp \
    dump/raw/eval1/wav.scp
# Evaluate log-F0 RMSE
./pyscripts/utils/evaluate_f0.py \
    exp/<model_dir_name>/<decode_dir_name>/eval1/wav/wav.scp \
    dump/raw/eval1/wav.scp
# If you want to calculate more precisely, limit the F0 range
./pyscripts/utils/evaluate_f0.py \
    --f0min xxx \
    --f0max yyy \
    exp/<model_dir_name>/<decode_dir_name>/eval1/wav/wav.scp \
    dump/raw/eval1/wav.scp
# Evaluate CER
./scripts/utils/evaluate_asr.sh \
    --model_tag <asr_model_tag> \
    --nj 1 \
    --inference_args "--beam_size 10 --ctc_weight 0.4 --lm_weight 0.0" \
    --gt_text "dump/raw/eval1/text" \
    exp/<model_dir_name>/<decode_dir_name>/eval1/wav/wav.scp \
    exp/<model_dir_name>/<decode_dir_name>/asr_results
# Since ASR model does not use punctuation, it is better to remove punctuations if it contains
./utils/remove_punctuation.pl < dump/raw/eval1/text > dump/raw/eval1/text.no_punc
./scripts/utils/evaluate_asr.sh \
    --model_tag <asr_model_tag> \
    --nj 1 \
    --inference_args "--beam_size 10 --ctc_weight 0.4 --lm_weight 0.0" \
    --gt_text "dump/raw/eval1/text.no_punc" \
    exp/<model_dir_name>/<decode_dir_name>/eval1/wav/wav.scp \
    exp/<model_dir_name>/<decode_dir_name>/asr_results
# Some ASR models assume the existence of silence at the beginning and the end of audio
# Then, you can perform silence padding with sox to get more reasonable ASR results
awk < "exp/<model_dir_name>/<decode_dir_name>/eval1/wav/wav.scp" \
    '{print $1 " sox " $2 " -t wav - pad 0.25 0.25 |"}' \
    > exp/<model_dir_name>/<decode_dir_name>/eval1/wav/wav_pad.scp
./scripts/utils/evaluate_asr.sh \
    --model_tag <asr_model_tag> \
    --nj 1 \
    --inference_args "--beam_size 10 --ctc_weight 0.4 --lm_weight 0.0" \
    --gt_text "dump/raw/eval1/text.no_punc" \
    exp/<model_dir_name>/<decode_dir_name>/eval1/wav/wav_pad.scp \
    exp/<model_dir_name>/<decode_dir_name>/asr_results
```

While these objective metrics can estimate the quality of synthesized speech, it is still difficult to fully determine human perceptual quality from these values, especially with high-fidelity generated speech.
Therefore, we recommend performing the subjective evaluation if you want to check perceptual quality in detail.

You can refer [this page](https://github.com/kan-bayashi/webMUSHRA/blob/master/HOW_TO_SETUP.md) to launch web-based subjective evaluation system with [webMUSHRA](https://github.com/audiolabs/webMUSHRA).

## Supported text frontend

You can change via `--g2p` option in `tts.sh`.

- `none`: Just separate by space
    - e.g.: `HH AH0 L OW1 <space> W ER1 L D` -> `[HH, AH0, L, OW1, <space>, W, ER1, L D]`
- `g2p_en`: [Kyubyong/g2p](https://github.com/Kyubyong/g2p)
    - e.g. `Hello World` -> `[HH, AH0, L, OW1, <space>, W, ER1, L D]`
- `g2p_en_no_space`: [Kyubyong/g2p](https://github.com/Kyubyong/g2p)
    - Same G2P but do not use word separator
    - e.g. `Hello World` -> `[HH, AH0, L, OW1, W, ER1, L, D]`
- `pyopenjtalk`: [r9y9/pyopenjtalk](https://github.com/r9y9/pyopenjtalk)
    - e.g. `こ、こんにちは` -> `[k, o, pau, k, o, N, n, i, ch, i, w, a]`
- `pyopenjtalk_kana`: [r9y9/pyopenjtalk](https://github.com/r9y9/pyopenjtalk)
    - Use kana instead of phoneme
    - e.g. `こ、こんにちは` -> `[コ, 、, コ, ン, ニ, チ, ワ]`
- `pyopenjtalk_accent`: [r9y9/pyopenjtalk](https://github.com/r9y9/pyopenjtalk)
    - Add accent labels in addition to phoneme labels
    - Based on [Developing a Japanese End-to-End Speech Synthesis Server Considering Accent Phrases](https://jglobal.jst.go.jp/detail?JGLOBAL_ID=202102244593559954)
    - e.g. `こ、こんにちは` -> `[k, 1, 0, o, 1, 0, k, 5, -4, o, 5, -4, N, 5, -3, n, 5, -2, i, 5, -2, ch, 5, -1, i, 5, -1, w, 5, 0, a, 5, 0]`
- `pyopenjtalk_accent_with_pause`: [r9y9/pyopenjtalk](https://github.com/r9y9/pyopenjtalk)
    - Add a pause label in addition to phoneme and accent labels
    - Based on [Developing a Japanese End-to-End Speech Synthesis Server Considering Accent Phrases](https://jglobal.jst.go.jp/detail?JGLOBAL_ID=202102244593559954)
    - e.g. `こ、こんにちは` -> `[k, 1, 0, o, 1, 0, pau, k, 5, -4, o, 5, -4, N, 5, -3, n, 5, -2, i, 5, -2, ch, 5, -1, i, 5, -1, w, 5, 0, a, 5, 0]`
- `pyopenjtalk_prosody`: [r9y9/pyopenjtalk](https://github.com/r9y9/pyopenjtalk)
    - Use special symbols for prosody control
    - Based on [Prosodic features control by symbols as input of sequence-to-sequence acoustic modeling for neural TTS](https://doi.org/10.1587/transinf.2020EDP7104)
    - e.g. `こ、こんにちは` -> `[^, k, #, o, _, k, o, [, N, n, i, ch, i, w, a, $]`
- `pypinyin`: [mozillanzg/python-pinyin](https://github.com/mozillazg/python-pinyin)
    - e.g. `卡尔普陪外孙玩滑梯。` -> `[ka3, er3, pu3, pei2, wai4, sun1, wan2, hua2, ti1, 。]`
- `pypinyin_phone`: [mozillanzg/python-pinyin](https://github.com/mozillazg/python-pinyin)
    - Separate into first and last parts
    - e.g. `卡尔普陪外孙玩滑梯。` -> `[k, a3, er3, p, u3, p, ei2, wai4, s, un1, uan2, h, ua2, t, i1, 。]`
- `espeak_ng_arabic`: [espeak-ng/espeak-ng](https://github.com/espeak-ng/espeak-ng)
    - e.g. `السلام عليكم` -> `[ʔ, a, s, s, ˈa, l, aː, m, ʕ, l, ˈiː, k, m]`
    - This result provided by the wrapper library [bootphon/phonemizer](https://github.com/bootphon/phonemizer)
- `espeak_ng_german`: [espeak-ng/espeak-ng](https://github.com/espeak-ng/espeak-ng)
    - e.g. `Das hört sich gut an.` -> `[d, a, s, h, ˈœ, ɾ, t, z, ɪ, ç, ɡ, ˈuː, t, ˈa, n, .]`
    - This result provided by the wrapper library [bootphon/phonemizer](https://github.com/bootphon/phonemizer)
- `espeak_ng_french`: [espeak-ng/espeak-ng](https://github.com/espeak-ng/espeak-ng)
    - e.g. `Bonjour le monde.` -> `[b, ɔ̃, ʒ, ˈu, ʁ, l, ə-, m, ˈɔ̃, d, .]`
    - This result provided by the wrapper library [bootphon/phonemizer](https://github.com/bootphon/phonemizer)
- `espeak_ng_spanish`: [espeak-ng/espeak-ng](https://github.com/espeak-ng/espeak-ng)
    - e.g. `Hola Mundo.` -> `[ˈo, l, a, m, ˈu, n, d, o, .]`
    - This result provided by the wrapper library [bootphon/phonemizer](https://github.com/bootphon/phonemizer)
- `espeak_ng_russian`: [espeak-ng/espeak-ng](https://github.com/espeak-ng/espeak-ng)
    - e.g. `Привет мир.` -> `[p, rʲ, i, vʲ, ˈe, t, mʲ, ˈi, r, .]`
    - This result provided by the wrapper library [bootphon/phonemizer](https://github.com/bootphon/phonemizer)
- `espeak_ng_greek`: [espeak-ng/espeak-ng](https://github.com/espeak-ng/espeak-ng)
    - e.g. `Γειά σου Κόσμε.` -> `[j, ˈa, s, u, k, ˈo, s, m, e, .]`
    - This result provided by the wrapper library [bootphon/phonemizer](https://github.com/bootphon/phonemizer)
- `espeak_ng_finnish`: [espeak-ng/espeak-ng](https://github.com/espeak-ng/espeak-ng)
    - e.g. `Hei maailma.` -> `[h, ˈei, m, ˈaː, ɪ, l, m, a, .]`
    - This result provided by the wrapper library [bootphon/phonemizer](https://github.com/bootphon/phonemizer)
- `espeak_ng_hungarian`: [espeak-ng/espeak-ng](https://github.com/espeak-ng/espeak-ng)
    - e.g. `Helló Világ.` -> `[h, ˈɛ, l, l, oː, v, ˈi, l, aː, ɡ, .]`
    - This result provided by the wrapper library [bootphon/phonemizer](https://github.com/bootphon/phonemizer)
- `espeak_ng_dutch`: [espeak-ng/espeak-ng](https://github.com/espeak-ng/espeak-ng)
    - e.g. `Hallo Wereld.` -> `[h, ˈɑ, l, oː, ʋ, ˈɪː, r, ə, l, t, .]`
    - This result provided by the wrapper library [bootphon/phonemizer](https://github.com/bootphon/phonemizer)
- `espeak_ng_hindi`: [espeak-ng/espeak-ng](https://github.com/espeak-ng/espeak-ng)
    - e.g. `नमस्ते दुनिया` -> `[n, ə, m, ˈʌ, s, t, eː, d, ˈʊ, n, ɪ, j, ˌaː]`
    - This result provided by the wrapper library [bootphon/phonemizer](https://github.com/bootphon/phonemizer)
- `espeak_ng_english_us_vits`: [espeak-ng/espeak-ng](https://github.com/espeak-ng/espeak-ng)
    - VITS official implementation-like processing (https://github.com/jaywalnut310/vits)
    - e.g. `Hello World.` -> `[h, ə, l, ˈ, o, ʊ, , <space>, w, ˈ, ɜ, ː, l, d, .]`
    - This result provided by the wrapper library [bootphon/phonemizer](https://github.com/bootphon/phonemizer)
- `g2pk`: [Kyubyong/g2pK](https://github.com/Kyubyong/g2pK)
    - e.g. `안녕하세요 세계입니다.` -> `[ᄋ, ᅡ, ᆫ, ᄂ, ᅧ, ᆼ, ᄒ, ᅡ, ᄉ, ᅦ, ᄋ, ᅭ,  , ᄉ, ᅦ, ᄀ, ᅨ, ᄋ, ᅵ, ᆷ, ᄂ, ᅵ, ᄃ, ᅡ, .]`
- `g2pk_no_space`: [Kyubyong/g2pK](https://github.com/Kyubyong/g2pK)
    - Same G2P but do not use word separator
    - e.g. `안녕하세요 세계입니다.` -> `[ᄋ, ᅡ, ᆫ, ᄂ, ᅧ, ᆼ, ᄒ, ᅡ, ᄉ, ᅦ, ᄋ, ᅭ, ᄉ, ᅦ, ᄀ, ᅨ, ᄋ, ᅵ, ᆷ, ᄂ, ᅵ, ᄃ, ᅡ, .]`
- `korean_jaso`: [jdongian/python-jamo](https://github.com/jdongian/python-jamo)
    - e.g. `나는 학교에 갑니다.` -> `[ᄂ, ᅡ, ᄂ, ᅳ, ᆫ, <space>, ᄒ, ᅡ, ᆨ, ᄀ, ᅭ, ᄋ, ᅦ, <space>, ᄀ, ᅡ, ᆸ, ᄂ, ᅵ, ᄃ, ᅡ, .]`
- `korean_jaso_no_space`: [jdongian/python-jamo](https://github.com/jdongian/python-jamo)
    - e.g. `나는 학교에 갑니다.` -> `[ᄂ, ᅡ, ᄂ, ᅳ, ᆫ, ᄒ, ᅡ, ᆨ, ᄀ, ᅭ, ᄋ, ᅦ, ᄀ, ᅡ, ᆸ, ᄂ, ᅵ, ᄃ, ᅡ, .]`

You can see the code example from [here](https://github.com/espnet/espnet/blob/cd7d28e987b00b30f8eb8efd7f4796f048dc3be9/test/espnet2/text/test_phoneme_tokenizer.py).

## Supported text cleaner

You can change via `--cleaner` option in `tts.sh`.

- `none`: No text cleaner.
- `tacotron`: [keithito/tacotron](https://github.com/keithito/tacotron)
    - e.g.`"(Hello-World);  & jr. & dr."` ->`HELLO WORLD, AND JUNIOR AND DOCTOR`
- `jaconv`: [kazuhikoarase/jaconv](https://github.com/kazuhikoarase/jaconv)
    - e.g. `”あらゆる”　現実を　〜　’すべて’ 自分の　ほうへ　ねじ曲げたのだ。"` -> `"あらゆる" 現実を ー \'すべて\' 自分の ほうへ ねじ曲げたのだ。`

You can see the code example from [here](https://github.com/espnet/espnet/blob/cd7d28e987b00b30f8eb8efd7f4796f048dc3be9/test/espnet2/text/test_cleaner.py).

## Supported Models

You can train the following models by changing `*.yaml` config for `--train_config` option in `tts.sh`.

### Single speaker model

- [Tacotron 2](https://arxiv.org/abs/1712.05884)
- [Transformer-TTS](https://arxiv.org/abs/1809.08895)
- [FastSpeech](https://arxiv.org/abs/1905.09263)
- [FastSpeech2](https://arxiv.org/abs/2006.04558) ([FastPitch](https://arxiv.org/abs/2006.06873))
- [Conformer](https://arxiv.org/abs/2005.08100)-based FastSpeech / FastSpeech2
- [VITS](https://arxiv.org/abs/2106.06103)

You can find example configs of the above models in [`egs2/ljspeech/tts1/conf/tuning`](../../ljspeech/tts1/conf/tuning).

### Multi speaker model extension

You can use / combine the following embedding to build multi-speaker model:
- [X-Vector](https://ieeexplore.ieee.org/abstract/document/8461375)
- [GST](https://arxiv.org/abs/1803.09017)
- Speaker ID embedding (One-hot vector -> Continuous embedding)
- Language ID embedding (One-hot vector -> Continuous embedding)

X-Vector is provided by kaldi and pretrained with VoxCeleb corpus.
You can find example configs of the above models in:
- [`egs2/vctk/tts1/conf/tuning`](../../vctk/tts1/conf/tuning).
- [`egs2/libritts/tts1/conf/tuning`](../../vctk/libritts/conf/tuning).
