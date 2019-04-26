import fire
import os
import subprocess
import numpy as np
import pandas as pd

from config import Configuration as config


class DataReader(object):
    def __init__(self, testing=False):

        # before starting anything, check if the right folder where we will
        # store data exists, otherwise create it
        if not os.path.exists(config.experimentsfolder):
            os.makedirs(config.experimentsfolder)

        if testing is True:
            self._test_data = self._load_data(testing)
        else:
            self._train_data = self._load_data()
            self._labels = self._load_labels()

    @property
    def train(self):
        return self._train_data

    @property
    def labels(self):
        return self._labels

    @property
    def test(self):
        return self._test_data

    # channels corresponding to the columns of <position>_motion.txt files
    # ordered according to the SHL dataset documentation.
    channels = {
        # [...]
        2: 'Acc_x',
        3: 'Acc_y',
        4: 'Acc_z',
        5: 'Gyr_x',
        6: 'Gyr_y',
        7: 'Gyr_z',
        8: 'Mag_x',
        9: 'Mag_y',
        10: 'Mag_z',
        11: 'Ori_w',
        12: 'Ori_x',
        13: 'Ori_y',
        14: 'Ori_z',
        15: 'Gra_x',
        16: 'Gra_y',
        17: 'Gra_z',
        18: 'LAcc_x',
        19: 'LAcc_y',
        20: 'LAcc_z',
        21: 'Pre'
        # [...]
    }

    modalities = [
        'Acc',
        'Gyr',
        'Mag',
        'LAc',
        'Gra',
        'Ori',
        'Pre'
    ]

    coarselabel_map = {
        0: 'null',
        1: 'still',
        2: 'walk',
        3: 'run',
        4: 'bike',
        5: 'car',
        6: 'bus',
        7: 'train',
        8: 'subway',
    }

    finelabel_map = {
        1: 'till;Stand;Outside',
        2: 'Still;Stand;Inside',
        3: 'Still;Sit;Outside',
        4: 'Still;Sit;Inside',
        5: 'Walking;Outside',
        6: 'Walking;Inside',
        7: 'Run',
        8: 'Bike',
        9: 'Car;Driver',
        10: 'Car;Passenger',
        11: 'Bus;Stand',
        12: 'Bus;Sit',
        13: 'Bus;Up;Stand',
        14: 'Bus;Up;Sit',
        15: 'Train;Stand',
        16: 'Train;Sit',
        17: 'Subway;Stand',
        18: 'Subway;Sit',
    }

    smartphone_positions = [
        'Torso',
        'Hips',
        'Bag',
        'Hand'
    ]

    trainfiles = {
        'User1': ['220617', '260617', '270617'],
        'User2': ['140617', '140717', '180717'],
        'User3': ['030717', '070717', '140617'],
    }

    testfiles = {
    }

    num_channels = len(channels)  # 20
    num_modalities = len(modalities)  # 7
    num_coarselabels = len(coarselabel_map)
    num_finelabels = len(finelabel_map)

    def _load_data(self):
        """
        Synopsis

         Returns
        """
        m = 23
        data = {}

        for user, days in self.trainfiles.items():
            data[user] = {}
            for day in days:
                data[user][day] = {}
                for position in self.smartphone_positions:
                    src = os.path.join(
                        config.datafolder,
                        'SHLDataset_preview_v1',
                        user,
                        day,
                        position + '_Motion.txt')

                    pipe = subprocess.Popen(
                        "wc -l < " + src,
                        shell=True,
                        stdout=subprocess.PIPE).stdout
                    n = pipe.read()

                    key = \
                        user + '_' +\
                        day + '_' +\
                        position

                    dest = os.path.join(
                        config.experimentsfolder, key + '.mmap')

                    data[user][day][position] = self._mmap_file(
                        src,
                        dest,
                        dtype=np.double,
                        shape=(int(n), m))

        return data

    def _load_labels(self):
        """
        Synopsis

         Returns
        """
        # n = 16310  # number of samples
        m = 8  # number of columns according to SHL dataset documentation
        filename = 'Label.txt'

        labels = {}

        for user, days in self.trainfiles.items():
            labels[user] = {}
            for day in days:

                src = os.path.join(
                    config.datafolder,
                    'SHLDataset_preview_v1',
                    user,
                    day,
                    filename)

                pipe = subprocess.Popen(
                    "wc -l < " + src,
                    shell=True,
                    stdout=subprocess.PIPE).stdout
                n = pipe.read()
                print(n)

                key = \
                    user + '_' + \
                    day + '_' + \
                    filename

                dest = os.path.join(
                    config.experimentsfolder, key + '.mmap')

                labels[user][day] = self._mmap_file(
                    src,
                    dest,
                    dtype=np.integer,
                    shape=(int(n), m))

        return labels

    def _mmap_file(self, src, dest, dtype, shape):
        if os.path.exists(dest):
            # just load mmap file contents
            print('%s exists, loading ...' % dest)
            mmap = np.memmap(
                dest,
                mode='r+',
                dtype=dtype,
                shape=shape)

            return mmap
        else:
            # build mmap file from scratch
            print('Building from scratch %s ...' % dest)
            print(shape)
            mmap = np.memmap(
                dest,
                mode='w+',
                dtype=dtype,
                shape=shape)

            chunksize = 5000
            offset = 0
            for chunk in pd.read_csv(src, delimiter=' ', chunksize=chunksize, header=None):
                mmap[offset:offset+chunk.shape[0]] = chunk.values
                offset += chunk.shape[0]

            return mmap


if __name__ == '__main__':
    # Python Fire is a library for automatically generating command line
    # interfaces (CLIs) from absolutely any Python object.
    fire.Fire(DataReader)
