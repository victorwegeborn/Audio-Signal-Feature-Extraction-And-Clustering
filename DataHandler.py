import os
import pickle
import librosa
import fnmatch
import argparse
import numpy as np
from time import time

import imp
import soundfile as sf
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# path to processed data. If no folder is present
# it will be created, as will its subdirectories.
PROCESSED_PATH = os.path.abspath('./buckets/')


# Features available for extraction
# Add here if more are implemented
DEFAULT_FEATURES = [
    'brightness',
    'mfcc',
    'rolloff',
    'decay'
]

class DataHandler:
    def __init__(self):
        self._summary = { 'targets': [], 'total': 0, 'longest_frame_size': 0 }

    def load_data(self, classes, features):
        # Load wanted features and concatenate
        # data for clustering
        return_data = []
        return_targets = []
        for c in classes:
            current_class = None
            for feature in features:
                relative_path = '{}/{}_{}.pkl'.format(feature, c, feature)
                feature_dict, _ = self._restore_data(relative_path)
                if current_class is None:
                    current_class = feature_dict['data']
                else:
                    current_class = np.concatenate((current_class, feature_dict['data']), axis=1)
            return_data.append(current_class)
            return_targets.append(np.zeros((current_class.shape[0],1), dtype=int))

        # Assign labels/targets to corresponting data class
        # Only used for evaluation of model
        for idx, targets in enumerate(return_targets):
            targets += idx

        # TODO: mergings of classes..

        # Pack features into training data
        data = None
        targets = None
        for idx, _ in enumerate(return_data):
            if data is None:
                data = return_data[idx]
                targets = return_targets[idx]
            else:
                data = np.concatenate((data, return_data[idx]))
                targets = np.concatenate((targets, return_targets[idx]))

        return data, targets


    def _wav_to_data(self, data_path):
        # Walk throught all subdirectories of data_path
        # Bundles all subdirs of a class into one bucket
        #       e.g cymbals/* == class 'cymbals'
        # Prints summary of samples in the end
        print(80 * '_')
        print('>> Samples-to-buckets phase')
        print('class\t\tnsamples\tshortest\tlongest')
        data_dicts = []
        subdirs = os.listdir(os.path.abspath(data_path))
        absdir = os.path.abspath(data_path)
        for subdir in subdirs:
            t0 = time()
            data_dict = {
                'wave_data': [],
                'sample_rates': [],
                'frame_sizes': [],
                'target': subdir,
                'max': 0,
                'min': 0,
                'nsamples': 0,
                'longest_frame_size': 0
            }
            for dirpath, _, files in os.walk(os.path.join(absdir, subdir)):
                for filename in fnmatch.filter(files, '*.wav'):
                    abs_filepath = os.path.join(dirpath, filename)
                    data, samplerate = sf.read(abs_filepath)
                    data_dict['wave_data'].append(data)
                    data_dict['sample_rates'].append(samplerate)
                    data_dict['frame_sizes'].append(data.shape[0])

            if len(data_dict['wave_data']) > 0:
                assert (np.array(data_dict['sample_rates']) == 44100).all()
                data_dict['max'] = max(data_dict['frame_sizes'])/44100
                data_dict['min'] = min(data_dict['frame_sizes'])/44100
                data_dict['nsamples'] = len(data_dict['wave_data'])

                print('%-9s\t%i\t\t%.2fs\t\t%.2fs' % (data_dict['target'],
                                                           data_dict['nsamples'],
                                                           data_dict['min'],
                                                           data_dict['max']))
                self._summary['targets'].append((data_dict['target'], data_dict['nsamples']))
                self._summary['total'] += data_dict['nsamples']
                self._summary['longest_frame_size'] = max(self._summary['longest_frame_size'], data_dict['max'])
                data_dicts.append(data_dict)

        print(80 * '_')
        print(">> Storing bucket files to disc:")
        for data_dict in data_dicts:
            info = self._store_data(data_dict['target'], data_dict)
            print('Saved at: {}'.format(info))

        # Print summary of data to console
        print('\n>> Summary:')
        print('target\t\tnsamples\tpercentage')
        for entry in self._summary['targets']:
            print('{:9s}\t{:d}\t\t{:.00%}'.format(entry[0], entry[1], entry[1]/self._summary['total']))
        print('total\t\t{:d}\t\t({:.0%})'.format(self._summary['total'], self._summary['total']/self._summary['total']))
        return data_dicts


    def _extract_data_features(self, feature_types, data_dicts):
        # Extracts features in feature_types list.
        # Each feature extraction is stored as dict like: {data: x, target: t}
        # Features are stored in a modular fasion for easier selection of data
        # to cluster on
        t0 = time()
        print(100 * '_')
        print('>> Feature extraction phase')
        for data_dict in data_dicts:
            for feature in feature_types:
                print('Extracting {} feature for {}'.format(feature, data_dict['target']))
                relative_path = feature + '/'
                feature_path = os.path.join(PROCESSED_PATH, relative_path)
                if not os.path.exists(feature_path):
                    os.makedirs(feature_path)

                data = eval('self._'+feature+'_extraction(data_dict)')
                self._store_data(relative_path+data_dict['target']+'_'+feature, data)
        print('>> Feature extraction completed in {:d}s'.format(t0 - time()))


    def _mfcc_extraction(self, data_dict):
        '''
        In speech recoginiton, MFCC's are the go to feature.
        I have very little knowlage of how they work, and i did not have the
        time to fully grasp what they are.

        https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
        http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
        '''

        # Window size and stride.
        # Initialy was set according to the link above (w:0.25, s:0.10), but was
        # experimentaly determined through out a couple of iterations.
        window_width = 0.220
        window_stride = 0.050
        feature_data = None
        for idx, data in enumerate(data_dict['wave_data']):
            mono_data = librosa.core.to_mono(data.T)
            mfcc11 = librosa.feature.mfcc(mono_data,
                                        sr=44100,
                                        n_mfcc=11,
                                        dct_type=2,
                                        hop_length=int(window_stride*44100),
                                        n_fft=int(window_width*44100))

            mfcc5 = librosa.feature.mfcc(mono_data,
                                        sr=44100,
                                        n_mfcc=5,
                                        dct_type=2,
                                        hop_length=int(window_stride*44100),
                                        n_fft=int(window_width*44100))

            mfcc2 = librosa.feature.mfcc(mono_data,
                                          sr=44100,
                                          n_mfcc=2,
                                          dct_type=2,
                                          hop_length=int(window_stride*44100),
                                          n_fft=int(window_width*44100))

            # Bundle up the mean and standard deviation of diffrent sets of mfccs.
            # This approach was a naive way to force same size data sets
            # for each sample, while extracting as much information as possible.
            mfcc_avg = np.concatenate((np.mean(mfcc2,axis=1), np.mean(mfcc11, axis=1), np.mean(mfcc5, axis=1)))
            mfcc_std = np.concatenate((np.std(mfcc2,axis=1), np.std(mfcc11, axis=1), np.std(mfcc5, axis=1)))
            mfcc_tot = np.concatenate((mfcc_avg, mfcc_std)).reshape(-1,1)

            if not isinstance(feature_data, np.ndarray):
                feature_data = np.zeros((data_dict['nsamples'], mfcc_tot.shape[0]))
            feature_data[idx,:,None] = mfcc_tot
        return {'data': feature_data, 'target': data_dict['target']}


    def _decay_extraction(self, data_dict):
        '''
        Extracts decay time from the max peak of the sample
        to some set threshold. Decay of a cymbal is generally
        longer then any Kick, Snare or Tom. Hence, it may be
        a good feature to separate cymbals from other parts of
        the drum sounds.
        '''

        window_time = 0.060#s
        window_frame_size = int(44100*window_time)
        threshold = 0.10
        feature_data = np.zeros((data_dict['nsamples'],1))
        for idx, data in enumerate(data_dict['wave_data']):
            # Convert from stereo to mono and
            # force peaks to be positive
            data = np.abs(np.mean(data, axis=1))

            # find top most peak i.e max transient
            max_peak, offset = np.max(data), np.argmax(data)

            # calculate number of frames from max peak
            nframes = data[offset:].shape[0]
            nwindows = int(np.floor(nframes/window_frame_size))
            maximized_window = np.zeros(data.shape)
            # Inside each window find maximum value i.e the peak of the window
            for i in range(nwindows):
                start = window_frame_size*i+offset
                end = window_frame_size*(i+1)+offset
                window = data[start:end]
                maximized_window[start:end] = np.max(window)
                # linearly interpolate between each window
                # in order to have a naive continous decay
                if i > 0:
                    prev_start = window_frame_size*(i-1)+offset
                    f = interp1d([prev_start, start], [maximized_window[prev_start], maximized_window[start]])
                    maximized_window[prev_start:start] = f(np.arange(prev_start, start))
                if i == nwindows-1:
                    f = interp1d([start, end], [maximized_window[start], 0])
                    maximized_window[start:end] = f(np.arange(start, end))

            # find number of frames from the peak
            # to where the decay is below some
            # threshold.
            decay_frames = 0
            for f in maximized_window[offset:]:
                decay_frames += 1
                if f < max_peak*threshold:
                    break

            # calculate decay time
            # and use as feature
            decay_time = decay_frames/44100.
            feature_data[idx] = decay_time

            if data_dict['target'] == 'cymbal' or data_dict['target'] == 'Kick':
                plt.plot(offset, max_peak, 'o')
                plt.plot(maximized_window)
                plt.plot(data)
                plt.show()
        return {'data': feature_data,'target': data_dict['target'] }


    def _brightness_extraction(self, data_dict):
        '''
        Find how much of the frequencys are above a set threshold.
        Cymbals have more energy in the higher part of the spectrum,
        while toms and kicks are lower.
        '''
        frequency_threshold = 1800
        feature_data = np.zeros((data_dict['nsamples'], 2))
        for idx, data in enumerate(data_dict['wave_data']):
            data = librosa.core.to_mono(data.T)
            fft = np.abs(np.fft.rfft(data))
            brightness = np.sum(fft[frequency_threshold:]) / np.sum(fft)
            feature_data[idx] = brightness
            '''
            if data_dict['target'] == 'cymbal':
                plt.plot(freqL, fL)
                plt.show()
            '''
        return {'data': feature_data, 'target': data_dict['target']}


    def _rolloff_extraction(self, data_dict):
        '''
        Extracts the frequency for which 85% of the enery is found.
        This approch was inspired by the spectral rolloff in the librosa
        package:
        https://librosa.github.io/librosa/generated/librosa.feature.spectral_rolloff.html
        '''
        feature_data = np.zeros((data_dict['nsamples'], 1))
        for idx, data in enumerate(data_dict['wave_data']):
            data = librosa.core.to_mono(data.T)
            fft = np.abs(np.fft.rfft(data))
            threshold = np.sum(fft)*0.85
            energy = 0.0
            for i in range(fft.size):
                energy += fft[i]
                if energy >= threshold:
                    feature_data[idx] = i
                    break

        return {'data': feature_data, 'target': data_dict['target']}


    def _store_data(self, filename, data_dict):
        absfilename = os.path.join(PROCESSED_PATH, filename) + '.pkl'
        with open(absfilename, 'wb') as f:
            pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL)
        return absfilename


    def _restore_data(self, filename):
        absfilename = os.path.join(PROCESSED_PATH, filename)
        with open(absfilename, 'rb') as f:
            data_dict = pickle.load(f)
        return data_dict, absfilename



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--data_path',
        help='Folder with .wav samples',
        required=True
    )
    parser.add_argument(
        '-f',
        '--feature_list',
        help='List of features to extract e.i -f <feature> -f ... \nfeatures={all, mfcc, decay, brightness, rolloff}',
        action='append',
        required=True
    )
    args = parser.parse_args()

    data = []
    if not os.path.exists(PROCESSED_PATH):
        os.makedirs(PROCESSED_PATH)

    # Turn all wave samples into class buckets according to their subdirectories
    # Stores data in PROCESSED_PATH for later extraction of features
    # NOTE: if bucketing has to be redone then remove files in PROCESSED_PATH
    if args.data_path:
        files = [f for f in os.listdir(PROCESSED_PATH) if f.endswith('.pkl')]
        if files:
            print(80 * '_')
            print('>> Loading bucket phase')
            for file in files:
                data_dict, path = DataHandler()._restore_data(file)
                data.append(data_dict)

                print('Loaded bucket from: {}'.format(path))
        else:
            data = DataHandler()._wav_to_data(args.data_path)

    # Extract data into features specified with args -f <feature>
    if len(args.feature_list) == 1 and args.feature_list[0] == 'all':
        DataHandler()._extract_data_features(DEFAULT_FEATURES, data)
    elif args.feature_list:
        DataHandler()._extract_data_features(args.feature_list, data)


if __name__ == '__main__':
    main()
