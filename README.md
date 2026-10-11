# HeRI-Nav

**Trajectory-Consistent Residual Learning for Underwater INS/DVL Navigation**

Shuyu Li (李书宇)  
Faculty of Information Science and Engineering, Ocean University of China

HeRI-Nav learns horizontal displacement corrections around an INS/DVL estimator.
The local residual is transported by the nominal interval-start heading and
accumulated into position. Source training constrains mission-average and
initially low-error-subset risks. During inference, sensor vectors and attitude
are jointly expressed in a source-fitted roll/pitch basis, preserving the
initial yaw and world-frame vectors. This projection adds no learned parameters.

This release provides the frozen all-source **seed-42** mean predictor, its
PyTorch architecture, embedded normalization, source calibration, and an offline
inference interface. Each trajectory uses one network. The manuscript's reported
means summarize scalar errors over three independent training seeds.

## Quick start

```sh
git clone https://github.com/BBOOOKK/HeRI-Nav-Code.git
cd HeRI-Nav-Code
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py validate --output validation.json
python run.py predict --input examples/synthetic_windows.npz --output prediction.npz
```

The example is a **synthetic interface test**, not an experimental trajectory or
accuracy benchmark. Validation compares computed outputs with saved predictions.
This release was checked with Python 3.14, PyTorch 2.14.0, and NumPy 2.5.3.

## Input and output

Supply an NPZ with these arrays:

| Array | Shape | Meaning |
|---|---|---|
| `x` | float32, N × 200 × 14 | N complete, nonoverlapping 8 s windows sampled at 25 Hz |
| `nominal` | N × 2 | Nominal horizontal endpoint positions, metres |
| `yaw` | N | Nominal interval-start headings, radians |
| `stamps` | N + 1 | Interval-boundary timestamps, nanoseconds |

Channel order is acceleration (3, m/s²), angular rate (3, rad/s), body-frame
DVL velocity (3, m/s), sample interval (s), roll and pitch (rad), and sine/cosine
of propagated yaw. `config.json` lists the channels. The runner first applies the source-basis projection and then converts the
yaw pair to window-relative heading; source standardization is embedded in the model.
Sensor features use the frame-converted measurements before the source-fitted
sensor calibration. Supply nominal positions and headings from the calibrated
physical estimator in a common navigation frame.

Prediction writes local residuals `mu` (metres), corrected endpoint `position`
(metres), and the input `stamps`. Correction accumulation starts at zero for
each file, so pass consecutive windows together to carry correction history.
The input already contains the physical estimator's positions and headings;
this release executes the learned correction stage.

## Model and files

- `model.pt`: frozen TorchScript mean predictor; 1,261,250 parameters.
- `model.py`: readable PyTorch encoder and mean-predictor definitions;
  `load_model()` loads the same weights and normalization buffers.
- `run.py`: input processing, inference, frame transport, and accumulation.
- `source_calibration.json`: the original source-fitted calibration constants.
- `config.json`: channel order, model identity, and original checkpoint hashes.
- `manifest.json`: SHA-256 file checks and the synthetic fixture identity.

The trained network has 1,261,637 parameters, including the covariance projection
used in pretraining. Navigation uses the mean output. Training data, full physical
filter replay, and reproduction of the multi-seed tables are outside this
inference release; the paper specifies their experimental settings.

## Citation

Use the repository's `CITATION.cff` for this software release. The associated
manuscript is in preparation; no publication DOI is assigned here.

Version 0.1.1 adds the source-fitted input projection used in the current
manuscript. The checkpoint weights are unchanged from v0.1.0. The fixed roll/pitch
center is recorded in `config.json`; it is applied equally to every input file.
Training risks were constrained before this inference-only transformation.
