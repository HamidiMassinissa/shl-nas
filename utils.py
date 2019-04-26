import os
import numpy as np

from config import Configuration as config


def save(obj, what, identifier=0):

    filename = "{}.{}.{}.{}-{}-{}-model-{}.sav".format(
        what,
        config.VERSION,
        config.REVISION,
        config.MINOR_REVISION,
        config.POSITION,
        config.USER,
        identifier
    )

    f = os.path.join(
        config.experimentsfolder,
        'confusion',
        filename
    )

    np.save(f, obj)
    return f
