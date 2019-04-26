"""
This file contains code tu train models individually
"""
import os
import numpy as np
import timeit
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix

import dataset
import model
import pipeline
import utils
import scoring

from config import Configuration

# get command line arguments in the form of a class with static attributes
config = Configuration()
config.parse_commandline(is_testing=True)


def trainSingleModel():
    """ Building the predictive model using previously found best
        hyperparameters.
    """

    print('==================================================')
    print('Training a unique model with the instantiation:')
    print(config.cmd_args)
    print('Experiment version %s.%s.%s-%s-%s'
          % (config.VERSION
             , config.REVISION
             , config.MINOR_REVISION
             , config.POSITION
             , config.USER))
    print('==================================================')

    # Build data reader and get training data
    dr = dataset.DataReader()
    X = dr.train
    y = dr.labels

    # Build pipeline
    p = pipeline.Pipeline(X, y)

    X = p.X
    y = p.y
    offsets_assoc = p.offsets_assoc
    modalities_assoc = p.modalities_assoc

    kernel_sizes = dict.fromkeys(modalities_assoc['Torso'])
    for i, (k, _) in enumerate(kernel_sizes.items()):
        offset = (i * 3)
        kernel_sizes[k] = config.kernel_sizes_list[offset:offset+3]

    """
# -----------------------------
    X_train, X_test, y_train, y_test =\
        train_test_split(
            X, y,
            test_size=0.30,
            random_state=1)
# -----------------------------
    """

    # Build estimator
    estimator = model.Estimator(
        num_classes=config.num_classes,
        num_modalities=dataset.DataReader.num_modalities,
        fingerprint_size=p.num_features,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate, # 0.1,
        decay=10**-1,
        num_filters=config.num_filters,
        kernel_sizes=kernel_sizes,
        overlap=config.overlap,
        num_units_dense_layer=config.num_units_dense_layer,
        dropout=config.dropout,
        offsets_assoc=offsets_assoc,
        modalities_assoc=modalities_assoc
    )

    estimator.__setstate__(estimator.__dict__)

    if config.VALIDATION == 'meta-segmented-cv':
        print('======================================================')
        print('       META-SEGMENTED CROSS-VALIDATION                ')
        print('======================================================')

        from metacvpartition import metacvpartition

        _X = X
        _y = y[:, 0]

        mxval = metacvpartition(
            _y
            , config.xval_nfolds
            , config.xval_metasegmentlength
            , debug=False)

        start = timeit.default_timer()  # -----
        y_pred = cross_val_predict(
            estimator,
            _X, _y,
            cv=mxval.splitsGenerator(),
            n_jobs=config.N_JOBS_cv,
            verbose=6)
        stop = timeit.default_timer()  # -----
        print('training time')
        print(stop - start)

        print(y_pred.shape)
        conf_mat = confusion_matrix(_y, y_pred)
        print(conf_mat)
        f = utils.save(conf_mat, 'confusion_matrix', estimator._io._num_persistor)
        print('confusion matrix saved to %s' % f)
        f = utils.save(y_pred, 'y_pred', estimator._io._num_persistor)
        print('y_pred saved to %s' % f)
        f = utils.save(_y, 'y', estimator._io._num_persistor)
        print('y saved to %s' % f)
        f = utils.save(list(mxval.splitsGenerator()), 'splits', estimator._io._num_persistor)
        print('Generated splits saved to %s' % f)

        # compute a bunch of scores -----------
        fscore_micro = f1_score(_y, y_pred, average='micro')
        print('[objective] f1_score_micro = %s%%' % (fscore_micro * 100,))
        fscore = scoring.Fscore(_y, y_pred, list(mxval.splitsGenerator()), debug=True)
        avg = fscore.avg_fscore()
        print('[objective] f1_score_avg = %s%%' % (avg * 100,))
        prre = fscore.prre_fscore()
        print('[objective] f1_score_prre = %s%%' % (prre * 100,))
        tpfp = fscore.tpfp_fscore()
        print('[objective] f1_score_tpfp = %s%%' % (tpfp * 100,))

        print('OK')


if __name__ == '__main__':
    best_hp = [176, 384, 0.1, 20, 10, 13, 10, 0.5588342062133987, 0.927376170245294, 0.9556776934579977, 0.6495583749193767]
    trainSingleModel()
