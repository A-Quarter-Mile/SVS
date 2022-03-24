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
    * [Single speaker model](#single-speaker-model)
    * [Multi speaker model](#multi-speaker-model)


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
- [Distributed training](https://espnet.github.io/espnet/espnet2_distributed.html)

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
│   └── token_list/   
│        └── phn_none  # token list
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
    ├── svs_stats_raw_phn_none # statistics
    └── svs_train_raw_phn_none # model
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
    spks: 5  # Number of speakers
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

While these objective metrics can estimate the quality of synthesized speech, it is still difficult to fully determine human perceptual quality from these values, especially with high-fidelity generated speech.
Therefore, we recommend performing the subjective evaluation if you want to check perceptual quality in detail.

You can refer [this page](https://github.com/kan-bayashi/webMUSHRA/blob/master/HOW_TO_SETUP.md) to launch web-based subjective evaluation system with [webMUSHRA](https://github.com/audiolabs/webMUSHRA).
```

## Supported text frontend

You can change via `--g2p` option in `svs.sh`.

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

You can train the following models by changing `*.yaml` config for `--train_config` option in `run.sh`.

### Single speaker model

- [Naive-RNN]()
- [GLU-Transformer]()
- [MLP-Singer](https://arxiv.org/abs/2106.07886)
- [XiaoIce](https://arxiv.org/pdf/2006.06261)

You can find example configs of the above models in [`egs/ofuton_p_utagoe_db/svs1/conf/tuning`](../../ofuton_p_utagoe_db/svs1/conf/tuning).

### Multi speaker model

You can use / combine the following embedding to build multi-speaker model:
- [X-Vector](https://ieeexplore.ieee.org/abstract/document/8461375)
- [GST](https://arxiv.org/abs/1803.09017)
- Speaker ID embedding (One-hot vector -> Continuous embedding)
- Language ID embedding (One-hot vector -> Continuous embedding)

X-Vector is provided by kaldi and pretrained with VoxCeleb corpus.
You can find example configs of the above models in:
- [`egs2/vctk/tts1/conf/tuning`](../../vctk/tts1/conf/tuning).
- [`egs2/libritts/tts1/conf/tuning`](../../vctk/libritts/conf/tuning).


