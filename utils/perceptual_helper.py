# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exports helper functions for perceptual evaluation
metrics.
"""

import numpy as np
import pesq

SAMPLE_RATE = 16000

def pesq_metric(reference, degraded):
    """Wrapper function for the pesq perceptual error metric.

    Args:
        reference: The clean reference file
        degraded: The distorted version of the reference file.

    Returns:
        The perceptual error as computed by the wrapped function.
    """

    reference = np.array(reference)
    degraded = np.array(degraded)

    degraded = degraded[0:len(reference)]
    return pesq.pesq(SAMPLE_RATE, reference, degraded, 'wb')
