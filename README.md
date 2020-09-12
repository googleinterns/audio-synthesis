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

### Experiments
The experiments folder contains all the experiment scripts. There are three main folders:
 - Representation Study: These experiments pertain to the experimenal application of GANs
    to audio synthesis using various signal representations.
 - Conditional: These experiments are for the conditional GANs models, using auxiliry historical
    data and MIDI information.
 - Basis Functions: 


### Results


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
