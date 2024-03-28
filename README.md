# Tacotron2-ZS (Tacotron2 based Zero-shot Voice Cloning)

PyTorch implementation of "Zero-Shot Voice Cloning Text-to-Speech for Dysphonia Disorder Speakers". 

![Alignment, Predicted Mel Spectrogram, Target Mel Spectrogram](tensorboard.png)


## Pre-requisites
1. NVIDIA GPU + CUDA cuDNN

## Setup
This implementation includes **distributed** and **automatic mixed precision** support
Distributed and Automatic Mixed Precision support relies on NVIDIA's [Apex] and [AMP].

1. Clone this repo: `git clone https://github.com/TitiAffandi/tacotron2-ZS.git`
2. CD into this repo: `cd tacotron2`
3. Initialize submodule: `git submodule init; git submodule update`
4. Install [PyTorch 1.0]
5. Install [Apex]
6. Install python requirements or build docker image 
    - Install python requirements: `pip install -r requirements.txt`

## Dataset 
1. Training dataset: LibriTTS-Train360 that can be downloaded from [LibriTTS dataset](https://www.openslr.org/resources/60/)
2. Testing dataset: UncommonVoice dataset that can be requested from [UncommonVoice](https://merriekay.com/uncommonvoice)

## Training
1. `python train_ZS.py --output_directory=outdir --log_directory=logdir`
2. (OPTIONAL) `tensorboard --logdir=outdir/logdir`

## Inference demo
1. Download our published [Tacotron 2] model
2. Download our published [WaveGlow] model
3. `jupyter notebook --ip=127.0.0.1 --port=31337`
4. Load inference.ipynb 

N.b.  When performing Mel-Spectrogram to Audio synthesis, make sure Tacotron 2
and the Mel decoder were trained on the same mel-spectrogram representation. 


## Acknowledgements
This implementation uses code from the following repos: [Rafael Valle](https://github.com/NVIDIA/tacotron2), [Keith
Ito](https://github.com/keithito/tacotron/), [Prem Seetharaman](https://github.com/pseeth/pytorch-stft), [Eren Golge](https://github.com/mozilla/TTS) as described in our code.


[pytorch 1.0]: https://github.com/pytorch/pytorch#installation
[Apex]: https://github.com/nvidia/apex
[AMP]: https://github.com/NVIDIA/apex/tree/master/apex/amp
