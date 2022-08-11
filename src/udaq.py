import numpy as np
import sys
import time
from pathlib import Path
import configparser
import csv
from math import floor
from itertools import groupby

from src.picoscope_5000a import PicoScope5000A, INPUT_RANGES

class udaq():

    def __init__(self):

        self._CHANNELS = ['A', 'B', 'C', 'D']

        self._num_events = 0
        self._num_traces = 0

        self._coupling = {'A': 'DC', 'B': 'DC', 'C': 'DC', 'D': 'DC'}
        self._polarity = {'A': 1, 'B': 1, 'C': 1, 'D': 1}
        self._is_enabled = {'A': False, 'B': False, 'C': False, 'D': False}
        self._is_trigger_enabled = {'A': False, 'B': False,
                                    'C': False, 'D': False}
        self._trigger_type = {'A': 'LEVEL', 'B': 'LEVEL',
                              'C': 'LEVEL', 'D': 'LEVEL'}
        self._trigger_direction = {'A': 'RISING', 'B': 'RISING',
                                   'C': 'RISING', 'D': 'RISING'}
        self._range = {'A': 10, 'B': 10, 'C': 10, 'D': 10}
        self._is_baseline_correction_enabled = {'A': False, 'B': False,
                                                'C': False, 'D': False}
        self._threshold = {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}
        self._timing = {'A': 'PEAK', 'B': 'PEAK', 'C': 'PEAK', 'D': 'PEAK'}
        self._traces = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        self._pulse_width = {'A': 10, 'B': 10, 'C': 10, 'D': 10}

        self._timebase = 0
        self._sample_interval = 0
        self._pre_trigger_window = 0.
        self._post_trigger_window = 0.
        self._pre_samples = 0
        self._post_samples = 0
        self._num_samples = 0
        self._num_captures = 0

        self._t_start_run = 0
        self._elapsed_time = 0
        self._run_time = 0

        self._num_runs = 1
        self._output_path = None
        self._run_number = 0
        self._output_filename = None
        self._output_file = None
        self._trace_filename = None
        self._trace_file = None
        self._trace_file_isopen = False

        if len(sys.argv) == 2:
            self._config_filename = Path.cwd()/str(sys.argv[1])
        else:
            self._config_filename = Path.cwd()/'udaq.config'
        
        self._config = None
        self._read_configuration()

        self.scope = PicoScope5000A()

    def _read_configuration(self):
        print(f'Configuration file: {self._config_filename}')
        self._config = configparser.ConfigParser(allow_no_value=True)
        self._config.read(str(self._config_filename))

        path = self._config['Run']['Output Path']
        if path == '.':
            self._output_path = Path.cwd()
        else:
            self._output_path = Path.cwd()/path
        self._run_time = self._config.getfloat('Run', 'Run Time')
        if self._config.has_option('Run','Number of Runs'):
            self._num_runs = self._config.getint('Run', 'Number of Runs')

        self._pre_trigger_window = self._config.getfloat('Sampling',
                                                         'Pre-Trigger Window')
        self._post_trigger_window \
            = self._config.getfloat('Sampling','Post-Trigger Window')
        self._timebase = self._config.getint('Sampling', 'Time Base')
        self._num_captures = self._config.getint('Sampling',
                                                 'Captures Per Block')

        for ch in self._CHANNELS:
            sec = 'Channel ' + ch
            self._is_enabled[ch] = sec in self._config.sections()
            if self._is_enabled[ch]:
                self._coupling[ch] = self._config[sec]['Coupling']
                self._polarity[ch] \
                    = np.sign(self._config.getint(sec, 'Polarity'))
                self._range[ch] = self._config.getfloat(sec, 'Range')
                self._is_baseline_correction_enabled[ch] \
                    = self._config[sec].getboolean('Baseline Correction')
                self._timing[ch] = self._config[sec]['Timing']
                self._is_trigger_enabled[ch] \
                    = self._config[sec].getboolean('Trigger Enabled')
                self._trigger_type[ch] = self._config[sec]['Trigger Type']
                self._trigger_direction[ch] \
                    = self._config[sec]['Trigger Direction']
                self._threshold[ch] = self._config[sec].getfloat('Threshold')
                if self._config.has_option(sec,'Traces'):
                    self._traces[ch] = self._config[sec].getint('Traces')
                if self._config.has_option(sec,'Pulse Width'):
                    self._pulse_width[ch] \
                        = self._config[sec].getint('Pulse Width')

    def _set_channels(self):
        for ch in self._CHANNELS:
            if self._is_enabled[ch]:
                self.scope.set_channel(ch, self._coupling[ch],
                                       self._range[ch], 0)

    def _set_sampling(self):
        dt = self.scope.get_interval_from_timebase(
            self._timebase)
        self._pre_samples  = floor(self._pre_trigger_window / 1e-9 / dt)
        self._post_samples = floor(self._post_trigger_window / 1e-9 / dt) + 1
        self._num_samples  = self._pre_samples + self._post_samples
        self._sample_interval = dt

    def _set_triggers(self):
        self.scope.set_advanced_triggers(self._is_trigger_enabled,
            self._trigger_type, self._trigger_direction, self._threshold, )

    def _open_output_file(self):
        self._output_filename =  self._output_path / 'Run{0:04d}.csv'\
                                                       .format(self._run_number)

        try:
             self._output_file = open(self._output_filename, 'w',
                                      newline='')
        except IOError:
            print('Error: Unable to open: {}'\
                  .format(self._output_filename))
            return 0

        writer = csv.writer(self._output_file)
        header = []
        for ch in self._CHANNELS:
            if self._is_enabled[ch]:
                header.append('time_'+ch)
                header.append('pulse_height_'+ch)
        writer.writerow(header)

    def _open_trace_file(self):
        self._trace_filename = self._output_path / 'Run{0:04d}_traces.csv'\
                                                       .format(self._run_number)
        try:
            self._trace_file = open(self._trace_filename, 'w',
                                    newline='')
        except IOError:
            print('Error: Unable to open: {}'\
                  .format(self._trace_filename))
            return 0
        self._trace_file_isopen = True
    
    def _process_data(self, data):
        x = data['x']
        times, baselines, pulseheights = [], [], []
        bl_samples = int(self._pre_samples * .8)
        for ch in self._CHANNELS:
            if self._is_enabled[ch]:
                channel_data = data[ch]
                if self._timing[ch] == 'PEAK':
                    ts = x[np.argmax(channel_data*self._polarity[ch], axis=1)]
                else: # ZERO or LEVEL
                      #     (https://stackoverflow.com/questions/23289976/
                      #       how-to-find-zero-crossings-with-hysteresis)
                    hi = channel_data >= self._threshold[ch]
                    if self._timing[ch] == 'ZERO':
                        lo = channel_data <= -self._threshold[ch]
                    else: # lEVEL
                        lo = channel_data <= self._threshold[ch]
                    if self._trigger_direction[ch] == 'FALLING':
                        y = hi[:, 1:]<hi[:, :-1]
                        z = lo[:, 1:]>lo[:, :-1]
                    else: # RISING
                        y = lo[:, 1:]<lo[:, :-1]
                        z = hi[:, 1:]>hi[:, :-1]
                    # Place a True at the end of the sampling period to handle
                    # failure to find a crossing.
                    fail = np.zeros(np.shape(y), dtype=bool)
                    fail[:,-1] = True
                    into  = np.where(y, y, fail)
                    outof = np.where(z, z, fail)
                    # Index of the *first* True (fail = last).
                    i_in = np.argmax(into, axis=1)
                    i_out = np.argmax(outof, axis=1)
                    ts = x[i_out]
                times.append(ts)
                channel_data *= self._polarity[ch]
                if self._is_baseline_correction_enabled[ch] and bl_samples > 0:
                    bl = channel_data[:, :bl_samples].mean(axis=1)
                else:
                    bl = np.zeros(len(channel_data))
                ph = (channel_data.max(axis=1) - bl)*1e3
                pulseheights.append(ph)
        #times = np.array(times) # This is redundant.
        return np.array(times), np.array(pulseheights)

    def _process_traces(self, data):
        x = data['x']
        for ch in self._CHANNELS:
            if self._is_enabled[ch] and self._traces[ch] > 0:
                channel_data = data[ch]
                evt = self._num_events + 1
                for trace in channel_data:
                    grad = np.gradient(trace)
                    # Find groups of consecutive identical values in an array.
                    # https://stackoverflow.com/questions/6352425
                    grouped_grad = [(k, sum(1 for i in g)) for k,g in groupby(grad>0)]

                    # Identify pulses within grouped_gradient list.
                    signals = [(gg[0] == True and gg[1] > self._pulse_width[ch]) for gg in grouped_grad]
                    number_of_signals = sum(signals)

                    if number_of_signals >= self._traces[ch]:
                        # Indices of items in the trace at the beginning of each group
                        # in grouped_gradient list.
                        items = [gg[1] for gg in grouped_grad]
                        indices = np.cumsum(items)

                        # Find the first pulse in the trace.
                        iSignals = np.where(signals)[0] - 1
                        i1 = indices[iSignals[0]]
                        w1 = items[iSignals[0]+1]
                        if w1 == 0:
                            continue
                        baseline1    = np.mean(trace[i1-5:i1-1])
                        pulseheight1 = np.max(trace[i1:i1+w1])-baseline1

                        # Find the largest additional pulse in the trace.
                        maxPh = 0
                        w2 = 0
                        pulseheight2 = 0
                        for j in range(1, number_of_signals, 1):
                            ii = indices[iSignals[j]]
                            ww = items[iSignals[j]+1]
                            if ww == 0:
                                continue
                            bl = np.mean(trace[ii-5:ii-1])
                            ph = np.max(trace[ii:ii+ww])-bl
                            if ph > maxPh:
                                i2 = ii
                                w2 = ww
                                pulseheight2 = ph
                                maxPh = ph
                        if w2 == 0:
                            continue
                        
                        if (self._traces[ch] == 1) or \
                            ((self._traces[ch] == 2) and (i2 - i1 > w1) \
                             and (pulseheight2 > np.abs(self._threshold[ch]))):
                            #((self._traces[ch] == 2) and (i2 - i1 > w1 + w2)):
                            self._write_trace(ch, evt, x, trace*1e3) # mV
                            self._num_traces += 1
                evt += 1

    def _write_trace(self, ch, evt, t, trace):
        if not self._trace_file_isopen:
            self._open_trace_file()
        self._trace_file.write(f'{ch},{evt}\n')
        # Use better precision than 12 bits to be safe.
        np.savetxt(self._trace_file, [t], delimiter=',', fmt='%.5e')
        np.savetxt(self._trace_file, [trace], delimiter=',', fmt='%.5e')

    def _write_output(self, times, pulseheights):
        writer = csv.writer(self._output_file, quoting=csv.QUOTE_NONE)
        table = []
        # There are data for each enabled channel.
        for i in range(len(times)):
            table.append(times[i])
            table.append(pulseheights[i])
        # Organize into rows by event
        for row in zip(*table):
            writer.writerow(row)

    def _close_output_file(self):
        try:
            self._output_file.close()
        except IOError:
            print('Error: Unable to close: {}'\
                  .format(self._output_filename))

    def _close_trace_file(self):
        try:
            self._trace_file.close()
        except IOError:
            print('Error: Unable to close: {}'\
                  .format(self._trace_filename))
        self._trace_file_isopen = False

    def _write_info_file(self):
        info_filename = self._output_filename.with_suffix('.info')
        try:
            info_file = open(info_filename, 'w', newline='', encoding="utf-8")
        except IOError:
            print(f'Error: Unable to open: {info_filename}\n')
        info_file.write(f'# Configuration File: {str(self._config_filename)}\n')
        info_file.write('\n[Run]\n')
        info_file.write(f'Start Time: {time.ctime(self._t_start_run)}\n')
        info_file.write(f'Run Time (s): {self._elapsed_time:.1f}\n')
        info_file.write(f'Events: {self._num_events}\n')
        info_file.write(f'Traces: {self._num_traces}\n')

        info_file.write('\n[Sampling]\n')
        info_file.write('Pre-Trigger Window (s): {0:.2e}\n'\
                        .format(self._pre_trigger_window))
        info_file.write('Post-Trigger Window (s): {0:.2e}\n'\
                        .format(self._post_trigger_window))
        info_file.write(f'Time Base: {self._timebase}\n')
        info_file.write(f'Sample Interval (ns): {self._sample_interval}\n')
        info_file.write(f'Samples Per Capture: {self._num_samples}\n')
        info_file.write(f'Captures Per Block: {self._num_captures}\n')

        for ch in self._CHANNELS:
            if self._is_enabled[ch]:
                info_file.write(f'\n[Channel {ch}]\n')
                info_file.write(f'Coupling: {self._coupling[ch]}\n')
                info_file.write(f'Polarity: {self._polarity[ch]}\n')
                info_file.write(f'Range: {self._range[ch]}\n')
                info_file.write('Baseline Correction: {0}\n'\
                    .format(self._is_baseline_correction_enabled[ch]))
                info_file.write(f'Timing: {self._timing[ch]}\n')
                info_file.write('Trigger Enabled: {0}\n'\
                    .format(self._is_trigger_enabled[ch]))
                info_file.write('Trigger Type: {0}\n'\
                    .format(self._trigger_type[ch]))
                info_file.write('Trigger Direction: {0}\n'\
                    .format(self._trigger_direction[ch]))
                info_file.write('Threshold (V): {0:.4f}\n'\
                    .format(self._threshold[ch]))
                sec = 'Channel ' + ch
                if self._config.has_option(sec,'Traces'):
                    info_file.write('Traces: {0:d}\n'\
                    .format(self._traces[ch]))
                if self._config.has_option(sec,'Pulse Width'):
                    info_file.write('Pulse Width: {0:d}\n'\
                    .format(self._pulse_width[ch]))

        info_file.close()

    def _acquire_run(self):

        for x in Path(self._output_path).glob('*.csv'):
            if x.is_file() and x.name[0:3] == 'Run':
                self._run_number = int(x.name[3:7]) + 1
        self._open_output_file()

        print(f'\nRun{self._run_number:04d}')

        self._t_start_run = time.time()
        self._elapsed_time = 0
        self._num_events = 0
        self._num_traces = 0
        while self._elapsed_time < self._run_time:
            self.scope.start_run(self._pre_samples, self._post_samples,
                                 self._timebase, self._num_captures)
            self.scope.wait_for_data()

            keys = ['x', 'A', 'B', 'C', 'D']
            x, [A, B, C, D] = self.scope.get_data()
            data = dict(zip(keys, [x, A, B, C, D]))
            times, pulseheights = self._process_data(data)

            self._write_output(times, pulseheights)
            
            self._process_traces(data)

            self._num_events += self._num_captures

            self._elapsed_time = time.time() - self._t_start_run

            print('Elapsed time: {0:.1f} s / {1:0.1f} s  | {2} events | {3} traces\r'\
                .format(self._elapsed_time, self._run_time, self._num_events, self._num_traces),
                end = '')

        self._close_output_file()

        if self._trace_file_isopen:
            self._close_trace_file()
            
        self._write_info_file()

    def acquire(self):

        self._set_channels()
        self._set_sampling()
        self._set_triggers()
        self.scope.set_up_buffers(self._num_samples, self._num_captures)

        if self._num_runs > 1:
            print(f'Acquiring {self._num_runs} runs.')
        for run in range(self._num_runs):
            self._acquire_run()


def main():
    daq = udaq()
    daq.acquire()


if __name__ == '__main__':
    main()
