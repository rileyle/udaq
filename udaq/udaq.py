import numpy as np
import sys
import time
from pathlib import Path
import configparser
import csv
from math import floor

from udaq.picoscope_5000a import PicoScope5000A, INPUT_RANGES

class udaq():

    def __init__(self):

        self._CHANNELS = ['A', 'B', 'C', 'D']

        self._num_events = 0

        self._coupling =  {'A': 'DC', 'B': 'DC', 'C': 'DC', 'D': 'DC'}
        self._is_enabled = {'A': False, 'B': False, 'C': False, 'D': False}
        self._is_trigger_enabled = {'A': False, 'B': False,
                                    'C': False, 'D': False}
        self._trigger_type = {'A': 'Level', 'B': 'Level',
                              'C': 'Level', 'D': 'Level'}
        self._trigger_direction = {'A': 'Rising', 'B': 'Rising',
                                   'C': 'Rising', 'D': 'Rising'}
        self._range = {'A': 10, 'B': 10, 'C': 10, 'D': 10}
        self._is_baseline_correction_enabled = {'A': False, 'B': False,
                                                'C': False, 'D': False}
        self._threshold = {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}

        self._timebase = 0
        self._sample_interval = 0
        self._pre_trigger_window = 0.
        self._post_trigger_window = 0.
        self._pre_samples = 0
        self._post_samples = 0
        self._num_samples = 0
        self._num_captures = 0

        self._t_start_run = 0
        self._run_time = 0

        self._output_path = None
        self._run_number = 0
        self._output_filename = None
        self._output_file = None

        if len(sys.argv) == 2:
            self._config_filename = Path.cwd()/str(sys.argv[1])
        else:
            self._config_filename = Path.cwd()/'udaq.config'

        self._read_configuration()

        self.scope = PicoScope5000A()

    def _read_configuration(self):
        print(f'Configuration file: {self._config_filename}')
        config = configparser.ConfigParser(allow_no_value=True)
        config.read(str(self._config_filename))

        path = config['Run']['Output Path']
        if path == '.':
            self._output_path = Path.cwd()
        else:
            self._output_path = Path.cwd()/path
        self._run_time = config.getfloat('Run', 'Run Time')

        self._pre_trigger_window = config.getfloat('Sampling',
                                                    'Pre-Trigger Window')
        self._post_trigger_window = config.getfloat('Sampling',
                                                    'Post-Trigger Window')
        self._timebase = config.getint('Sampling', 'Time Base')
        self._num_captures = config.getint('Sampling',
                                           'Captures Per Block')

        for ch in self._CHANNELS:
            sec = 'Channel ' + ch
            self._is_enabled[ch] = sec in config.sections()
            if self._is_enabled[ch]:
                self._coupling[ch] = config[sec]['Coupling']
                self._range[ch] = config.getfloat(sec, 'Range')
                self._is_baseline_correction_enabled[ch] \
                    = config[sec].getboolean('Baseline Correction')
                self._is_trigger_enabled[ch] \
                    = config[sec].getboolean('Trigger Enabled')
                if self._is_trigger_enabled[ch]:
                    self._trigger_type[ch] = config[sec]['Trigger Type']
                    self._trigger_direction[ch] \
                        = config[sec]['Trigger Direction']
                    self._threshold[ch] = config[sec].getfloat('Threshold')

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
        for x in Path(self._output_path).glob('*.csv'):
            if x.is_file() and x.name[0:3] == 'Run':
                self._run_number = int(x.name[3:7]) + 1

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
                header.append('trigger_time_'+ch)
                header.append('pulse_height_'+ch)
        writer.writerow(header)

    def _process_data(self, data):
        x = data['x']
        times, baselines, pulseheights = [], [], []
        bl_samples = int(self._pre_samples * .8)
        for ch in self._CHANNELS:
            if self._is_enabled[ch]:
                channel_data = data[ch]
                if self._is_baseline_correction_enabled[ch] and bl_samples > 0:
                    bl = channel_data[:, :bl_samples].mean(axis=1)
                else:
                    bl = np.zeros(len(channel_data))
                ph = (channel_data.max(axis=1) - bl)*1e3
                ts = x[np.argmax(channel_data, axis=1)]
                times.append(ts)
                pulseheights.append(ph)
        times = np.array(times)
        pulseheights = np.array(pulseheights)

        writer = csv.writer(self._output_file)
        table = []
        for i in range(len(times)):
            if self._is_enabled[self._CHANNELS[i]]:
                table.append(times[i])
                table.append(pulseheights[i])
        for row in zip(*table):
            writer.writerow(row)

    def _close_output_file(self):
        try:
            self._output_file.close()
        except IOError:
            print('Error: Unable to close: {}'\
                  .format(self._output_filename))

    def _write_info_file(self):
        info_filename = self._output_filename.with_suffix('.info')
        try:
            info_file = open(info_filename, 'w', newline='', encoding="utf-8")
        except IOError:
            print(f'Error: Unable to open: {info_filename}\n')
        info_file.write(f'Start Time: {time.ctime(self._t_start_run)}\n')
        info_file.write(f'Run Time: {self._run_time:.1f} s\n')
        for ch in self._CHANNELS:
            if self._is_enabled[ch]:
                info_file.write(f'Channel {ch}:\n')
                info_file.write(f'    Coupling: {self._coupling[ch]}\n')
                info_file.write(f'    Range: {self._range[ch]}\n')
                info_file.write('    Baseline Correction: {0}\n'\
                    .format(self._is_baseline_correction_enabled[ch]))
                if self._is_trigger_enabled[ch]:
                    info_file.write('    Trigger:\n')
                    info_file.write('        Trigger Type: {0}\n'\
                        .format(self._trigger_type[ch]))
                    info_file.write('        Trigger Direction: {0}\n'\
                        .format(self._trigger_direction[ch]))
                    info_file.write('        Threshold: {0:.2f} V\n'\
                        .format(self._threshold[ch]))

        info_file.write('Pre-Trigger Window: {0:.2f} s\n'\
                        .format(self._pre_trigger_window))
        info_file.write('Post-Trigger Window: {0:.2f} s\n'\
                        .format(self._post_trigger_window))
        info_file.write(f'Time Base: {self._timebase}\n')
        info_file.write(f'Sample Interval: {self._sample_interval} ns\n')
        info_file.write(f'Samples Per Capture: {self._num_samples}\n')
        info_file.write(f'Captures Per Block: {self._num_captures}\n')
        info_file.close()

    def acquire_run(self):

        self._set_channels()

        self._set_sampling()

        self._set_triggers()

        self.scope.set_up_buffers(self._num_samples, self._num_captures)

        self._open_output_file()

        self._t_start_run = time.time()
        elapsed_time = 0
        while elapsed_time < self._run_time:
            self.scope.start_run(self._pre_samples, self._post_samples,
                                 self._timebase, self._num_captures)
            self.scope.wait_for_data()

            keys = ['x', 'A', 'B', 'C', 'D']
            x, [A, B, C, D] = self.scope.get_data()
            self._process_data(dict(zip(keys, [x, A, B, C, D])))

            print(f'Run time: {elapsed_time:.1f} s\r', end = '')

            elapsed_time = time.time() - self._t_start_run

        self._run_time = elapsed_time

        self._close_output_file()

        self._write_info_file()

def main():
    daq = udaq()
    #daq._open_output_file()
    #daq._close_output_file()
    #daq._write_info_file()
    daq.acquire_run()


if __name__ == '__main__':
    main()
