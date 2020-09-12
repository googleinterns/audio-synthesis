# Audio Synthesis

This repository contains an exploration of generative models for audio synthesis.

**This is not an officially supported Google product.**

## Instructions
This section contains the instructions for setting up the cloned reposotory
and running experiments. **NOTE:** It is intended that all files are executed
root directory of this reposotory.

### Setup
This project is setup so that componet files can be included through the
audio_synthesis package. Hence, to execute scripts, you must ensure that the
directory containing this cloned repostory is in the Python path. This can
be done temporarly using the following command:

```export PYTHONPATH="${PYTHONPATH}:/my/other/path"```

### Datasets
This reposotory contains scripts for pre-processing two datasets, MAESTRO and SpeechMNIST.
These scripts are contained in the ```setup/``` folder.

For MAESTRO, the script ```setup/preprocess_maestro_with_midi_dataset.py``` processes the 
audio and algiend MIDI into chunks of a desired size. The script allows you to set how many
hours of data you wish to load as well as how to quantize the MIDI data. The process for
constructing the dataset is:
 - Download the dataset from https://magenta.tensorflow.org/datasets/maestro and unzip it. 
   - **NOTE:** If you only want a small amount of data then it is probably more efficient
   to download the MIDI separatly, extract all that (fast) and then only extract as much audio
   you need.
 - Configure the script:
   - APPROX_TOTAL_HOURS: How many hours to extract.
   - DATA_POINT_LENGTH: Length of the datapoints (in samples).
   - RAW_DATA_PATH: Path to the audio (.wav) files.
   - PROCESSED_DATA_PATH: Where to save the resulting .npz files
 - Execute preprocessing script.
 
For SpeechMNIST, the script ```setup/preprocess_speech_mnist_dataset.py``` constructs the dataset.
Follow the steps below:
 - Download the speech commands dataset from http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
 and extract it.
 - Specify RAW_DATA_PATH to point to the dataset location.
 - Execute the script.

### Experiments
The experiments folder contains all the experiment scripts. There are three main folders:
 - Representation Study: These experiments pertain to the experimenal application of GANs
    to audio synthesis using various signal representations.
 - Conditional: These experiments are for the conditional GANs models, using auxiliry historical
    data and MIDI information.
 - Basis Functions: The learned signal decomposition experiments.

**NOTE:** In all experiment scripts, there is a line of code at the top of ```main()```, that looks
like ```os.environ["CUDA_VISIBLE_DEVICES"] = '2'```. This is used to specify the visability of GPUs
to TensorFlow. Since, these programs are only configured for one GPU, this prevents TF from locking up
the others but leaving them unused.

All experiments typically specify all their paramaters up the top. To restore an experiment from a checkpoint,
add ```model.restore('ckpt-<number>', <num_elapsed_epochs>)``` just above ```model.train()```. Ensure that CHECKPOINT_DIR
is pointing to the correct location.

### Results
This folder contains scripts for extracting results from the trained models. The scripts are described below:
 - listening_evaluation_audio.py: Extracts audio files for a subjective listening evaluation from a collection
 of models defined in the script.
 - extract_basis_functions.py: Extracts the learned bases functions from specified models and plots them in the
 time and frequency domains.
 - snr_vs_pesq_graphs.py: Script for generating the theoretical robustness graphs, looking at SNR vs PESQ.

## Source Code Headers

Every file containing source code must include copyright and license
information. This includes any JS/CSS files that you might be serving out to
browsers. (This is to help well-intentioned people avoid accidental copying that
doesn't comply with the license.)

Apache header:

    Copyright 2020 Google LLC

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
