# ClearVoice

## 👉🏻[HuggingFace Space Demo](https://huggingface.co/spaces/alibabasglab/ClearVoice)👈🏻 | 👉🏻[ModelScope Space Demo](https://modelscope.cn/studios/iic/ClearerVoice-Studio)👈🏻

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Usage](#2-usage)

## 1. Introduction

ClearVoice offers a unified inference platform for `speech enhancement`, `speech separation`, and `audio-visual target speaker extraction`. It is designed to simplify the adoption of our pre-trained models for your speech processing purpose or the integration into your projects. Currently, we provide the following pre-trained models:

| Tasks (Sampling rate)                          | Models (HuggingFace Links)                                                                                                                                                                                                                                  |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Speech Enhancement (16kHz & 48kHz)             | `MossFormer2_SE_48K` ([link](https://huggingface.co/alibabasglab/MossFormer2_SE_48K)), `FRCRN_SE_16K` ([link](https://huggingface.co/alibabasglab/FRCRN_SE_16K)), `MossFormerGAN_SE_16K` ([link](https://huggingface.co/alibabasglab/MossFormerGAN_SE_16K)) |
| Speech Separation (16kHz)                      | `MossFormer2_SS_16K` ([link](https://huggingface.co/alibabasglab/MossFormer2_SS_16K))                                                                                                                                                                       |
| Audio-Visual Target Speaker Extraction (16kHz) | `AV_MossFormer2_TSE_16K` ([link](https://huggingface.co/alibabasglab/AV_MossFormer2_TSE_16K))                                                                                                                                                               |

You don't need to manually download the pre-trained models—they are automatically fetched during inference.

## 2. Usage

### Step-by-Step Guide

If you haven't created a Conda environment for ClearerVoice-Studio yet, follow steps 1 and 2. Otherwise, skip directly to step 3.

1. **Clone the Repository**

```sh
git clone https://github.com/mylukin/ClearerVoice-Studio.git
```

2. **Create Conda Environment**

```sh
cd ClearerVoice-Studio
conda create -n ClearerVoice-Studio python=3.11
conda activate ClearerVoice-Studio
pip install -r requirements.txt
```

It should also work for python 3.9, 3.10 and 3.12!

> **Note:** 在 ubuntu 和 windows 安装过程中，如果遇到关于 c++构建环境的前置安装以及 pip setuptools wheel 的工具更新问题，请自行手动安装解决 （感谢@RichardQin1）。

3. **Run Demo**

```sh
cd clearvoice
python demo.py
```

or

```sh
cd clearvoice
python demo_with_more_comments.py
```

- You may activate each demo case by setting to True in `demo.py` and `demo_with_more_comments.py`.
- Supported audio format: .flac .wav
- Supported video format: .avi .mp4 .mov .webm

4. **Use Scripts**

Use `MossFormer2_SE_48K` model for fullband (48kHz) speech enhancement task:

```python
from clearvoice import ClearVoice

myClearVoice = ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K'])

#process single wave file
output_wav = myClearVoice(input_path='samples/input.wav', online_write=False)
myClearVoice.write(output_wav, output_path='samples/output_MossFormer2_SE_48K.wav')

#process wave directory
myClearVoice(input_path='samples/path_to_input_wavs', online_write=True, output_path='samples/path_to_output_wavs')

#process wave list file
myClearVoice(input_path='samples/scp/audio_samples.scp', online_write=True, output_path='samples/path_to_output_wavs_scp')
```

Parameter Description:

- `task`: Choose one of the three tasks `speech_enhancement`, `speech_separation`, and `target_speaker_extraction`
- `model_names`: List of model names, choose one or more models for the task
- `input_path`: Path to the input audio/video file, input audio/video directory, or a list file (.scp)
- `online_write`: Set to `True` to enable saving the enhanced/separated audio/video directly to local files during processing, otherwise, the enhanced/separated audio is returned. (Only supports `False` for `speech_enhancement`, `speech_separation` when processing single wave file`)
- `output_path`: Path to a file or a directory to save the enhanced/separated audio/video file

这里给出了一个较详细的中文使用教程：https://stable-learn.com/zh/clearvoice-studio-tutorial

## 3. Model Performance

**Speech enhancement models:**
We evaluated our released speech enhancement models on the popular benchmarks: [VoiceBank+DEMAND](https://paperswithcode.com/dataset/demand) testset (16kHz & 48kHz) and [DNS-Challenge-2020](https://paperswithcode.com/dataset/deep-noise-suppression-2020) (Interspeech) testset (non-reverb, 16kHz). Different from the most published papers that tailored each model for each test set, our evaluation here uses unified models on the two test sets. The evaluation metrics are generated by [SpeechScore](https://github.com/modelscope/ClearerVoice-Studio/tree/main/speechscore).

**VoiceBank+DEMAND testset (tested on 16kHz)**
|Model |PESQ |NB_PESQ |CBAK |COVL |CSIG |STOI |SISDR |SNR |SRMR |SSNR |P808_MOS|SIG |BAK |OVRL |ISR |SAR |SDR |FWSEGSNR |LLR |LSD |MCD|
|----- |--- |------- |---- |---- |---- |---- |----- |--- |---- |---- |------ |--- |--- |---- |--- |--- |--- |-------- |--- |--- |---|
|Noisy |1.97 | 3.32 |2.79 |2.70 |3.32 |0.92 |8.44 |9.35 |7.81 |6.13 |3.05 |3.37 |3.32 |2.79 |28.11 |8.53 |8.44 |14.77 |0.78 |1.40 |4.15|
|FRCRN_SE_16K |3.23 | 3.86 |3.47 |**3.83**|4.29 |0.95 |19.22 |19.16 |9.21 |7.60 |**3.59**|3.46 |**4.11**|3.20 |12.66 |21.16 |11.71 |**20.76**|0.37 |0.98 |**0.56**|
|MossFormerGAN_SE_16K|**3.47**|**3.96**|**3.50**|3.73 |**4.40**|**0.96**|**19.45**|**19.36**|9.07 |**9.09**|3.57 |**3.50**|4.09 |**3.23**|25.98 |21.18 |**19.42**|20.20 |**0.34**|**0.79**|0.70|
|MossFormer2_SE_48K |3.16 | 3.77 |3.32 |3.58 |4.14 |0.95 |19.38 |19.22 |**9.61**|6.86 |3.53 |**3.50**|4.07 |3.22 |**12.05**|**21.84**|11.47 |16.69 |0.57 |1.72 |0.62|

**DNS-Challenge-2020 testset (tested on 16kHz)**
|Model |PESQ |NB_PESQ |CBAK |COVL |CSIG |STOI |SISDR |SNR |SRMR |SSNR |P808_MOS|SIG |BAK |OVRL |ISR |SAR |SDR |FWSEGSNR |LLR |LSD |MCD|
|----- |--- |------- |---- |---- |---- |---- |----- |--- |---- |---- |------ |--- |--- |---- |--- |--- |--- |-------- |--- |--- |---|
|Noisy |1.58 | 2.16 |2.66 |2.06 |2.72 |0.91 |9.07 |9.95 |6.13 |9.35 |3.15 |3.39 |2.61 |2.48 |34.57 |9.09 |9.06 |15.87 |1.07 |1.88 |6.42|
|FRCRN_SE_16K |3.24 | 3.66 |3.76 |3.63 |4.31 |**0.98**|19.99 |19.89 |8.77 |7.60 |4.03 |3.58 |4.15 |3.33 |**8.90** |20.14 |7.93 |**22.59**|0.50 |1.69 |0.97|
|MossFormerGAN_SE_16K|**3.57**|**3.88**|**3.93**|**3.92**|**4.56**|**0.98**|**20.60**|**20.44**|8.68 |**14.03**|**4.05**|**3.58**|**4.18**|**3.36**|8.88 |**20.81**|**7.98** |21.62 |**0.45**|**1.65**|**0.89**|
|MossFormer2_SE_48K |2.94 | 3.45 |3.36 |2.94 |3.47 |0.97 |17.75 |17.65 |**9.26**|11.86 |3.92 |3.51 |4.13 |3.26 |8.55 |18.40 |7.48 |16.10 |0.98 |3.02 |1.15|

**VoiceBank+DEMAND testset (tested on 48kHz)** (We included our evaluations on other open-sourced models using SpeechScore)
|Model |PESQ |NB_PESQ |CBAK |COVL |CSIG |STOI |SISDR |SNR |SRMR |SSNR |P808_MOS|SIG |BAK |OVRL |ISR |SAR |SDR |FWSEGSNR |LLR |LSD |MCD|
|----- |--- |------- |---- |---- |---- |---- |----- |--- |---- |---- |------ |--- |--- |---- |--- |--- |--- |-------- |--- |--- |---|
|Noisy |1.97 | 2.87 |2.79 |2.70 |3.32 |0.92 |8.39 |9.30 |7.81 |6.13 |3.07 |3.35 |3.12 |2.69 |33.75 |8.42 |8.39 |13.98 |0.75 |1.45 |5.41|
|MossFormer2_SE_48K |**3.15**|**3.77**|**3.33**|**3.64**|**4.23**|**0.95**|**19.36**|**19.22**|9.61 |7.03 |**3.53**| 3.41 |**4.10**|**3.15**|**4.08**|**21.23** |4.06 |14.45 |NA |1.86 |**0.53**|
|Resemble_enhance |2.84 | 3.58 |3.14 |NA |NA |0.94 |12.42 |12.79 |9.08 |7.07 |**3.53**|**3.42**| 3.99 |3.12 |13.62 |12.66 |10.31 |14.56 |1.50 |1.66 | 1.54 |
|DeepFilterNet |3.03 | 3.71 |3.29 |3.55 |4.20 |0.94 |15.71 |15.66 |**9.66**|**7.19**|3.47 |3.40 |4.00 |3.10 |28.01 |16.20 |**15.79**|**15.69**|**0.55**|**0.94**| 1.77 |

- Resemble_enhance ([Github](https://github.com/resemble-ai/resemble-enhance)) is an open-sourced 44.1kHz pure speech enhancement platform from Resemble-AI since 2023, we resampled to 48khz before making evaluation.
- DeepFilterNet ([Github](https://github.com/Rikorose/DeepFilterNet)) is a low complexity speech enhancement framework for Full-Band audio (48kHz) using on deep filtering.
  > **Note:** We observed anomalies in two speech metrics, LLR and LSD, after processing with the 48 kHz models. We will further investigate the issue to identify the cause.
