# EventVisionCNN

EventVisionCNN is a CNN classification system for event-camera (DVS) data that converts asynchronous per-pixel event 
streams from the NMNIST and DVS128 Gesture datasets into frames ready for convolutional neural network (CNN) training.
It uses multiple accumulation strategies for encoding (such as decay, voxel, etc.), and benchmarks their efficacy against
a conventional frame-based system (typical camera recording)

## Background
Rather than capturing frames like a typical camera, an event camera measures each pixel's brightness change with
high dynamic range and microsecond-level timing. Each pixel has its own circuit that watches the log of incoming light and fires
an event the moment that value moves beyond a threshold. The raw output is a stream of `(x, y, time, polarity)` tuples
as an Address-Event Representation (AER) rather than a 2D image.

## Installation
```bash
git clone https://github.com/VishalManikanden/eventvisioncnn.git
cd eventvisioncnn
pip install -e .
```

## Importing the Data
Datasets are not included in this repo; they are fetched via the `tonic` module

```python
import os
import certifi

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['SSL_CERT_DIR'] = os.path.dirname(certifi.where())

import tonic

# NMNIST dataset:
nmnist_train = tonic.datasets.NMNIST(save_to='./data', train=True)
nmnist_test = tonic.datasets.NMNIST(save_to='./data', train=False)

# DVS128 Gesture dataset:
tonic.datasets.DVSGesture.train_url = "https://ndownloader.figshare.com/files/38022171"
tonic.datasets.DVSGesture.test_url = "https://ndownloader.figshare.com/files/38020584"

gesture_train = tonic.datasets.DVSGesture(save_to='./data', train=True)
gesture_test = tonic.datasets.DVSGesture(save_to='./data', train=False)
```

## Sample Usage
See the example model training frameworks (such as `sample_nmnist.py`)

## Encoding Strategies
`events_to_frame()` supports 4 accumulation strategies for converting a raw event stream into a CNN-ready frame:

| Strategy      | Description                                                                                                                                             |
|---------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| `fixed_time`  | Accumulates all events within a fixed time window                                                                                                       |
| `fixed_count` | Accumulates a fixed number of consecutive events, with window duration self-adjusting to motion speed                                                   |
| `decay`       | Exponentially-weighted accumulation, with recent events having more weightage than older events (`tau` must be scaled to the recording's own timescale) |
| `voxel`       | Splits the time range into a specific number of bins, each with its own ON/OFF channel pair, preserving when within the recording an event happened     |

`baseline.events_to_conventional_frame()` simulates what a single conventional-camera exposure would capture: the same 
accumulation as `fixed_time`, but with the ON/OFF channels summed into one to discard the polarity information. This allows
for a benchmark comparison in accuracy between the CNNs trained on the conventional frames vs event-based frames

## Benchmark Results (CNN Trained on DVS128 Gesture, 11 Classes)
Evaluated under identical CNN architectures, dropout rates (0.4), and training configurations. The only variable was
the input encoding:

| Encoding                                       | Test Accuracy |
|------------------------------------------------|---------------|
| Voxel, 8 bins                                  | 0.769         |
| Fixed-time event frames                        | 0.746         |
| Voxel, 5 bins                                  | 0.746         |
| Conventional baseline (summed, single-channel) | 0.705         |

**All three event-based encodings outperformed the conventional single-channel baseline by ~4-6 percentage points (~9% increase).** See `benchmark_results.csv`
for the full results (including parameter counts and test loss). However, a separate comparison at a different dropout
setting showed that the polarity information alone (in fixed time vs the baseline) did not reliably outperform the baseline,
while voxel consistently did. This might mean that hyperparameter tuning or the structure of the CNN has just as large of
an impact as the encoding strategy itself.

## Running Tests
Runs the full test suite: synthetic checks on the encoding math, dataset shapes, degradation utilities, etc. None of the
tests require the NMNIST or DVS128 Gesture datasets to be downloaded first since they use test frames/arrays.
```bash
pytest eventvisioncnn/tests/
```

## Citations
```bibtex
@article{orchard2015converting,
  title={Converting static image datasets to spiking neuromorphic datasets using saccades},
  author={Orchard, Garrick and Jayawant, Ajinkya and Cohen, Gregory K and Thakor, Nitish},
  journal={Frontiers in neuroscience},
  year={2015},
  publisher={Frontiers}
}
 
@inproceedings{amir2017low,
  title={A low power, fully event-based gesture recognition system},
  author={Amir, Arnon and Taba, Brian and Berg, David and Melano, Timothy and McKinstry, Jeffrey and Di Nolfo, Carmelo and Nayak, Tapan and Andreopoulos, Alexander and Garreau, Guillaume and Mendoza, Marcela and others},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2017}
}
```

## License
MIT ([LICENSE](LICENSE))