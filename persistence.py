import os
import numpy as np

from config import Configuration as config

class Persistor(object):
    """
     Synopsis
        This class is reponsible of persisting various outputs of the neural
        networks that are tested in a given python session.
        Each persistor corrsponds to a given model (unique correspondance).
        `_num_persistor` is used to identify the outputs of a given model in a
        unique manner.
    """

    _num_persistor = -1

    def __init__(self):

        # before starting anything, check if the right folder where we will
        # store data exists, otherwise create it
        # files will be located in `./generated/5.2/3/`
        #           config.experimentsfolder ___|  |
        #                       config.REVISION ___|

        if not os.path.exists(config.persistencefolder):
            os.makedirs(config.persistencefolder)

        Persistor._num_persistor += 1

    def persist(self, name, value, epoch, training_step, max_training_steps):
        """
         Synopsis
            Stores outputs of a given component (layer) of the Tensorflow
            computational graph. Outputs of each epoch are stored in separate
            files.

            params:
                name            (str) name of the graph component
                value           (2d or 3d tensor) output of the graph component
                epoch           (int) training epoch
                trianing_step   (int) training step
        """

        filename = str(self._num_persistor) + '-' + str(epoch) + '-' + name + \
            '.' + config.VERSION + '.' + config.REVISION + '.mmap'
        dest = os.path.join(config.persistencefolder, filename)

        # debug
        print('training_step = %d' % training_step)
        print('shape of %s output is %s' % (name, value.shape,))

        if training_step == 0:
            assert not os.path.exists(dest)

            mmap = np.memmap(
                dest,
                mode='w+',  # Create or overwrite existing file for reading and writing.
                dtype=value.dtype,
                shape=(max_training_steps,) + value.shape
            )

            mmap[training_step] = value
            mmap.flush()

        else:
            mmap = np.memmap(
                dest,
                mode='r+',  # Open existing file for reading and writing.
                dtype=value.dtype,
                shape=(max_training_steps,) + value.shape
            )

            mmap[training_step] = value
            mmap.flush()
