import os
import timeit
import skopt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

import model
import dataset
import pipeline
import scoring
import utils
import hyperparameters

from config import Configuration as config

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

num_classes = config.num_classes
num_modalities = dataset.DataReader.num_modalities
batch_size = config.batch_size
num_features = p.num_features

space = hyperparameters.space


def objective(params):

    kernel_sizes = dict.fromkeys(modalities_assoc['Torso'])
    for i, (k, _) in enumerate(kernel_sizes.items()):
        offset = (i * 3) + 2
        kernel_sizes[k] = params[offset:offset+3]

    estimator = model.Estimator(
        num_classes=num_classes,
        num_modalities=num_modalities,
        fingerprint_size=num_features,
        batch_size=batch_size,
        learning_rate=params[0], # 0.1,
        decay=10**-1,
        num_filters=params[1],
        kernel_sizes=kernel_sizes,
        overlap=params[23],
        num_units_dense_layer=params[24],
        dropout=params[25],
        offsets_assoc=offsets_assoc,
        modalities_assoc=modalities_assoc
    )

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

        y_pred = cross_val_predict(
            estimator,
            _X, _y,
            cv=mxval.splitsGenerator(),
            n_jobs=config.N_JOBS_cv,
            verbose=6)

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
        fscore = scoring.Fscore(_y, y_pred, list(mxval.splitGenerator()))
        avg = fscore.avg_fscore()
        print('[objective] f1_score_avg = %s%%' % (avg * 100,))
        prre = fscore.prre_fscore()
        print('[objective] f1_score_prre = %s%%' % (prre * 100,))
        tpfp = fscore.tpfp_fscore()
        print('[objective] f1_score_tpfp = %s%%' % (tpfp * 100,))

        return -avg


def runBayes():
    """ Launch bayesian optimization in order to tune Tensorflow model's
        hyperparameter
    """
    print('==================================================')
    print('Bayesian optimization using Gaussian processes ...')
    print('Experiment version %s.%s.%s-%s-%s'
          % (config.VERSION
             , config.REVISION
             , config.MINOR_REVISION
             , config.POSITION
             , config.USER))
    print('==================================================')


    start = timeit.default_timer()  # -----------------
    r = skopt.gp_minimize(
        objective,
        space,
        n_calls=config.N_CALLS,
        random_state=config.SEED,
        n_jobs=config.N_JOBS_bayes,
        verbose=True)
    stop = timeit.default_timer()   # -----------------
    print('Bayesian Optimization took')
    print(stop - start)


    # save the model to disk
    f = os.path.join(
        # VERSION,  # folder
        # 'bayesOptResults' + MINOR_VERSION + '.' + MAJOR_VERSION + '.sav')
        config.experimentsfolder,
        'bayesOptResults.' \
        + config.VERSION \
        + '.' + config.REVISION \
        + '.' + config.MINOR_REVISION \
        + '-' + config.POSITION \
        + '-' + config.USER \
        + '-' + config.day_out \
        + '.sav')

    skopt.dump(r, open(f, 'wb'))

    print('OK')


if __name__ == '__main__':
    runBayes()
    # params = [50, 90, 90, 0.1, 18, 30, 0.6]
    # objective(params)
