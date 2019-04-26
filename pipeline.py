import fire
import os
import numpy as np
import pickle

from dataset import DataReader
from config import Configuration as config


class Pipeline(object):
    def __init__(self, X, y, testing=False):
        # 1. determine shape
        shape = self._get_shape(X)

        if self._loaded():
            # 2.a. load files
            X, y, offsets_assoc, modalities_assoc = \
                self._load_mmap_file(shape, create=False)

        else:
            # 2.b. create files and build lds, labels, assoc
            X, y, offsets_assoc, modalities_assoc = \
                self._build_learning_dataset(
                    X, y,
                    testing)

        X, y = self._clean_data(X, y)

        # 3. return them or store them as class attributes
        self._X = X
        self._y = y
        self._offsets_assoc = offsets_assoc
        self._modalities_assoc = modalities_assoc

    def _loaded(self):
        files = [
            'learning_dataset-',
            'labels-',
            'offsets_assoc-',
            'modalities_assoc-'
        ]

        check = True

        for f in files:
            # check existence
            filename = \
                f + \
                config.VERSION + '.' + \
                config.REVISION + \
                '.mmap'
            dest = os.path.join('generated', filename)
            check = check and os.path.exists(dest)

        return check

    @property
    def num_features(self):
        """ number of features that are outputted from this pipeline """
        return self._X.shape[2]

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    @property
    def offsets_assoc(self):
        """
         Returns an associative list or a dictionry which will correspond
         each user and day to a tuple containing the starting and ending
         indices in the tensor X.
        """
        return self._offsets_assoc

    @property
    def modalities_assoc(self):
        """
         Returns an associative list or a dictionry which will correspond
         each modality to a list of positions in the tensor X.
        """
        return self._modalities_assoc

    def _load_mmap_file(self, shape, create=False):
        """
        Synopsis

         Returns
        """
        num_sequences, num_samples, num_features = shape

        if create:
            mode = 'w+'
        else:
            mode = 'r+'

        # lds: learning dataset
        filename = \
            'learning_dataset-' + \
            config.VERSION + '.' + \
            config.REVISION + \
            '.mmap'
        dest = os.path.join('generated', filename)

        # build mmap file from scratch, It is REQUIRED!
        print('Building from scratch %s ... it may take a while' % dest)
        mmap = np.memmap(
            dest,
            mode=mode,
            dtype=np.double,
            shape=(
                num_sequences,
                num_samples,
                num_features)
        )

        # labels
        filename = \
            'labels-' + \
            config.VERSION + '.' + \
            config.REVISION + \
            '.mmap'
        dest = os.path.join('generated', filename)

        labels = np.memmap(
            dest,
            mode=mode,
            dtype=np.double,
            shape=(
                num_sequences,
                num_samples)
        )

        # offsets_assoc
        filename = \
            'offsets_assoc-' + \
            config.VERSION + '.' + \
            config.REVISION + \
            '.mmap'
        dest = os.path.join('generated', filename)

        if create:
            offsets_assoc = dest  # haha connard, tu as osé le faire
        else:
            mode = 'rb'
            with open(dest, mode) as f:
                offsets_assoc = pickle.load(f)

        # modalities_assoc
        filename = \
            'modalities_assoc-' + \
            config.VERSION + '.' + \
            config.REVISION + \
            '.mmap'
        dest = os.path.join('generated', filename)

        if create:
            modalities_assoc = dest  # haha connard, tu as osé le faire
        else:
            mode = 'rb'
            with open(dest, mode) as f:
                modalities_assoc = pickle.load(f)

        return mmap, labels, offsets_assoc, modalities_assoc

    def _get_shape(self, X):
        # determine shape of learning dataset
        width = 6000
        height = 0  # number of sequences (depends on the number of users and days)
        for user in DataReader.trainfiles.keys():
            days = X[user]

            for day, day_value in days.items():
                depth = 0  # number of channels (depends on the number of smartphones positions)
                for pos in DataReader.smartphone_positions:
                    discard = day_value[pos].shape[0] % 6000
                    reshaped = np.reshape(
                        day_value[pos][:-discard, 0],
                        (-1, 6000))

                    for num_chan, chan_name in DataReader.channels.items():
                        depth += 1

                height += reshaped.shape[0]

        return (height, width, depth)

    def _build_learning_dataset(self, X, y, testing=False):
        """
        Synopsis
         construct the learning dataset which will be stored as a memory
         mappeed object in order to leave program's heap alone ...

         Returns
        """
        # determine shape of learning dataset
        shape = self._get_shape(X)

        # allocate memory for learning dataset
        lds, \
            labels, \
            offsets_assoc_filename, \
            modalities_assoc_filename = \
            self._load_mmap_file(shape, create=True)

        offsets_assoc = {
            user: {
                day: []
                for day in DataReader.trainfiles[user]
            }
            for user in DataReader.trainfiles.keys()
        }

        modalities_assoc = {
            pos: {
                m: []
                for m in DataReader.modalities
            }
            for pos in DataReader.smartphone_positions
        }

        # fill learning dataset with data
        offset = 0
        for user in DataReader.trainfiles.keys():
            print(user)
            days = X[user]
            for day, day_value in days.items():  # all days
                index = 0
                for pos in DataReader.smartphone_positions:
                    for num_chan, chan_name in DataReader.channels.items():
                        discard = day_value[pos].shape[0] % 6000
                        reshaped = np.reshape(
                            day_value[pos][:-discard, num_chan],
                            (-1, 6000))

                        lds[offset:offset+reshaped.shape[0], :, index] = reshaped[:]

                        if offset == 0:  # fill this structure only in the first pass, i.e. offset == 0
                            modalities_assoc[pos][chan_name[:3]].append(index)

                        if config.DEBUG is True:
                            rand_x = np.random.randint(
                                low=offset,
                                high=offset+reshaped.shape[0])

                        print('oki')
                        index += 1

                # label = y[user][day]
                reshaped = np.reshape(
                    y[user][day][:-discard, 1],  # 1 corresponds to coarse labels wheareas 2 to fine labels
                    (-1, 6000))
                labels[offset:offset+reshaped.shape[0], :] = reshaped[:]

                if config.DEBUG is True:
                    rand_x = np.random.randint(
                        low=offset,
                        high=offset+reshaped.shape[0])

                offsets_assoc[user][day].append((offset, offset+reshaped.shape[0]))
                offset += reshaped.shape[0]

        # persist offsets_assoc
        with open(offsets_assoc_filename, 'wb+') as f:
            pickle.dump(offsets_assoc, f, pickle.HIGHEST_PROTOCOL)

        # persist offsets_assoc
        with open(modalities_assoc_filename, 'wb+') as f:
            pickle.dump(modalities_assoc, f, pickle.HIGHEST_PROTOCOL)

        return (lds, labels, offsets_assoc, modalities_assoc)

    def CHECK_for_nans(self, X):
        # checking for nan's
        nans = []
        for i, a in enumerate(X):
            if np.isnan(a).any():
                nans.append(i)

        return nans

    def CHECK_for_heterogeneous_labels(self, y):
        hetero = []
        for i, a in enumerate(y):
            if not np.all(a == a[0]):
                hetero.append(i)

        print(hetero)
        return hetero

    def CHECK_for_null_labels(self, y):
        null = []
        for i, a in enumerate(y):
            if np.all(a == 0):
                null.append(i)

        return null

    def _clean_data(self, X, y):
        """
        Synopsis
         replace sequences containing NaNs from X and from the corrsponding
         entries in y. These entries are replaced by entries of the null class
         that do not contain NaNs (sinon c'est le serpent qui se mord la queue).

         Returns
        """
        nans = self.CHECK_for_nans(X)
        if len(nans) == 0:
            print('[_clean_data] noting to clean')
            return X, y

        nulls = self.CHECK_for_null_labels(y)

        # remove nulls containing NaNs
        nulls = [ null for null in nulls if null not in nans ]

        for i, nan in enumerate(nans):
            X[nan] = X[nulls[i]]
            y[nan] = y[nulls[i]]

        # assert ------
        nans = self.CHECK_for_nans(X)
        assert len(nans) == 0
        # -------------

        return X, y


if __name__ == '__main__':
    # fire.Fire(SensorFusion)
    # fire.Fire(Pipeline)

    import dataset
    # ===> code from `bayesOpt.py`
    # Build data reader and get training data
    dr = dataset.DataReader()
    X = dr.train
    y = dr.labels

    # Build pipeline
    p = Pipeline(X, y)

    X = p.X
    y = p.y
    offsets_assoc = p.offsets_assoc
    modalities_assoc = p.modalities_assoc

    p.CHECK_for_nans(X)
    p.CHECK_for_heterogeneous_labels(y)
    p.CHECK_for_null_labels(y)
