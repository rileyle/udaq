# udaq

A command-line/Python interface for data acquisition from a 5000-series
PicoScope

The `./src/picoscope_5000a.py` interface is adopted from the
[gamma-spectroscopy](https://github.com/davidfokkema/gamma-spectroscopy)
package developed by David Fokkema and modified to incorporate advanced
triggering and four-channel PicoScopes.

---
## Setup and Installation

### Dependencies
- Python 3
- NumPy
- PicoSDK C libraries (see instructions in the README for the
  [PicoSDK Python wrappers](https://github.com/picotech/picosdk-python-wrappers))
- [PicoSDK Python wrappers](https://github.com/picotech/picosdk-python-wrappers)

### Install

```
pip install git+https://github.com/rileyle/udaq
```

## Run `udaq` from the command line:

```
udaq <configuration file name (relative path)>
```

The times and pulse heights from each active channel specified by the
configuration file are written to an ouput file in CSV format. configuration
parameters and details of the measurement are written to a `.info`
file.

## Configuration File

`udaq` uses the
[`configparser`](https://docs.python.org/3/library/configparser.html) package
to read acquisition parameters from a text file. (The `.info` file uses this
format to facilitate reading with a `configparser` as well.)

See the sample configuration file: `./tools/udaq.config`

### [Run]

`Output Path` : relative path from the current directory to the directory where
output files are written

`Run Time` : acquisition time in seconds

`Number of Runs` : number of runs to acquire

### [Sampling]

`Pre-Trigger Window` : time in seconds to sample before the trigger

`Post-Trigger Window` : time in seconds to sample after the trigger

`Time Base` : n where the sampling interval is (n–3) / 62,500,000 seconds
(n=4 is fastest)

`Captures Per Block` : number of events to collect in each rapid block

The dead time involved in retrieving each block can be minimized relative to
the total acquisition time by acquiring multiple events in each block. A good
starting value is 1000.

### [Channel A] ([Channel B], [Channel C], [Channel D])

(Include a section for each active channel.)

`Coupling` : `AC` or `DC`

`Range` : Range setting in volts (avalable ranges: `0.01`, `0.02`, `0.05`,
`0.1`, `0.2`, `0.5`, `1`, `2`, `5`, `10`, `20`). The channel will acquire in
the range &#177;<Range> V.

`Baseline Correction` : Apply a baseline correction to peak heights using the
average of the leading 80% of the pre-trigger samples (`True` or `False`)

`Timing` : Strategy used to determine times (`PEAK` or `ZERO`)

`Trigger Enabled` : Enable/disable triggering on this channel (`True` or
`False`)

`Trigger Type` : Trigger mode (only `LEVEL` is supported at present)

`Trigger Direction` : `RISING` or `FALLING` This value is used for triggering
and for `ZERO` timing.

`Threshold` : Trigger threshold in V. This value is used for triggering and
`ZERO` timing.

`Traces` : `1` : Write all traces, `2` : write only traces including two 
signals. Write time and voltage data from all enabled channels to a CSV 
file `RUNXXXX_traces.csv`. (See also `Pulse Width`)

`Pulse Width` : Width in samples of the minimum rise time of signals on
this channel for the purpose of identifying traces to write. (See also 
`Traces`.)

The `ZERO` timing strategy determines the time the signal leaves a window
between +`Threshold` and -`Threshold`. If `Trigger Direction` is `RISING`, the
time the signal crosses +`Threshold` is recorded, and if `Trigger Direction` is
`FALLING`, the time the signal crosses -`Threshold` is recorded.
