from math import ceil
import tensorflow as tf
from sklearn.base import BaseEstimator
import math
import fire
import numpy as np
from sklearn.model_selection import StratifiedKFold

from persistence import Persistor
from config import Configuration as config


tf.random.set_random_seed(config.SEED)


class Estimator(BaseEstimator):
    """ An implementation of the scikit-learn estimator interface
    """
    def __init__(self,
                 num_classes,
                 num_modalities,
                 fingerprint_size,
                 batch_size,
                 learning_rate,
                 decay,
                 num_filters,
                 kernel_sizes,
                 overlap,
                 num_units_dense_layer,
                 dropout,
                 offsets_assoc,
                 modalities_assoc):

        # info
        print('Estimator hyperparameters:')
        print('\tnum_classes=%d' % num_classes)
        print('\tnum_modalities=%d' % num_modalities)
        print('\tfingerprint_size=%d' % fingerprint_size)
        print('\tbatch_size=%d' % batch_size)
        print('\tlearning_rate=%f' % learning_rate)
        print('\tdecay=%f' % decay)
        print('\tnum_filters=%d' % num_filters)
        print('\tkernel_sizes=%s' % (kernel_sizes,))
        print('\toverlap=%f' % overlap)
        print('\tnum_units_dense_layer=%d' % num_units_dense_layer)
        print('\tdropout=%f' % dropout)
        print('\toffsets_assoc=%s' % (offsets_assoc,))
        print('\tmodalities_assoc=%s' % (modalities_assoc,))

        self.num_classes = num_classes
        self.num_modalities = num_modalities
        self.fingerprint_size = fingerprint_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.decay = decay
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.overlap = overlap
        self.num_units_dense_layer = num_units_dense_layer
        self.dropout = dropout
        self.offsets_assoc = offsets_assoc
        self.modalities_assoc = modalities_assoc

        self._io = Persistor()
        self.__dict__ = self._create_model(self.__dict__)

    def __setstate__(self, dict):
        """ This function is responsible of making the right calls in order to
            build the model. We get through this Python built-in function to
            build our model because of Pickling issues, in other words, this
            function is called when unpickeling is performed on the serialized
            object.
            for more information, take a look at:
                https://docs.python.org/2.0/lib/pickle-example.html
        """
        print('[__setstate__]')
        self.__dict__ = self._create_model(dict)

    def _step_decay(self, epoch):
        """ returns a new value for the learning rate dropped by a given factor
            at the construction of the model
        """
        lr = self.learning_rate * math.pow(self.decay, math.floor((1+epoch)/10))
        return lr

    def _train(self, X, y, epoch, saving):
        """
         Synopsis
            This function performs a complete pass through the training set.
        """
        # info
        print('[_train] epoch = %d' % epoch)

        # init
        lr = self.learning_rate
        if epoch % 2 == 0:
            lr = self._step_decay(epoch)
            # info
            print('[_train] new learning rate = %f' % lr)

        _unik, _cnt = np.unique(y, return_counts=True)

        training_step = 0
        total_accuracy = 0
        max_training_steps = int(X.shape[0] / self.batch_size + 1)

        # get batches (test folds) preserving the percentage of samples of each
        # class
        sfk = StratifiedKFold(max_training_steps, random_state=config.SEED)
        splits = sfk.split(np.zeros(X.shape[0]), y)

        for _, test_index in splits:

            X_b = X[test_index]
            y_b = y[test_index]

            evaluation_results = self._session.run(
                fetches=self.nodes_to_evaluate,
                feed_dict={
                    self._fingerprint_input: X_b,
                    self._ground_truth_input: y_b,
                    self._learning_rate_input: lr,
                    self._training: True,
                }
            )

            tf.logging.info(
                'Step #%d: rate %f, train accuracy %.5f%%, cross entropy %f'
                % (training_step,
                   evaluation_results['_learning_rate_input'],
                   evaluation_results['_evaluation_step'] * 100,
                   evaluation_results['_cross_entropy_mean']))

            total_accuracy += evaluation_results['_evaluation_step']
            training_step += 1

        # write changes to disk
        for k, v in self.nodes_to_persist.items():
            # info
            print('shape of %s = %s' % (k, v.shape))

            v.flush()

        return total_accuracy / training_step * 100

    def fit(self, X, y, epoch=0, saving=False):
        """ An implementation of the fitting function
        """
        print('[fit]')

        for epoch in range(config.max_training_epochs):
            train_accuracy = self._train(X, y, epoch, saving)
            print('[fit] total train accuracy = %f' % train_accuracy)

        return self

    def predict(self, X, close_session=True):
        """ An implementation of the predicting function
        """
        print('[predict]')

        logits_to_probs,\
            predicted_indices\
            = self._session.run([
                self._probs,
                self._predicted_indices
            ], feed_dict={
                self._fingerprint_input: X,
                self._training: False,
                # self._training: True,
            })

        print(logits_to_probs)

        if close_session:
            # close tensorflow default session
            self.close_session()

        return predicted_indices

    def close_session(self):
        self._session.close()

    def _get_seq_len(self, initial_size, kernel_sizes, overlap):
        # compute conv layers output shape
        # w = (w - k + 2p) / s + 1
        # take a look at: https://www.tensorflow.org/api_guides/python/nn#Convolution
        # and here also: https://stackoverflow.com/a/37674568/3441514

        p = initial_size
        for kernel_size in kernel_sizes:
            s = int(kernel_size * (1 - overlap))
            w = ceil( float( p - kernel_size + 1 ) / float(s) )
            p = int(w / 2)

        return p

    def _create_model(self, dict):
        """
         Synopsis
            This function is responsible of creating or building the
            machine learning model. Note that we are using a dictionary
            `dict` rather than the usual `self` for creating the class
            members because of Pickling-related issues which dictated this
            way of ...
        """
        num_classes = dict['num_classes']
        num_modalities = dict['num_modalities']
        fingerprint_size = dict['fingerprint_size']
        batch_size = dict['batch_size']
        num_filters = dict['num_filters']
        kernel_sizes = dict['kernel_sizes']
        overlap = dict['overlap']
        num_units_dense_layer = dict['num_units_dense_layer']
        dropout = dict['dropout']
        offsets_assoc = dict['offsets_assoc']
        modalities_assoc = dict['modalities_assoc']

        modalities = modalities_assoc[config.POSITION].keys()

        print('[_create_model]')

        tf.reset_default_graph()
        tf.logging.set_verbosity(tf.logging.INFO)

        _training = tf.placeholder(tf.bool)
        dict['_training'] = _training

        _fingerprint_input = tf.placeholder(
          # tf.float64,
          tf.float32,
          [None, None, fingerprint_size],
          name='fingerprint_input')
        dict['_fingerprint_input'] = _fingerprint_input

        # self._lengths_input = tf.placeholder(
        #     tf.float64,
        #     [None],
        #     name='lengths_input')

        _ground_truth_input = tf.placeholder(
            tf.int64,
            [None],
            name='groundtruth_input')
        dict['_ground_truth_input'] = _ground_truth_input

        _ground_truth_input_one_hot = tf.one_hot(
            _ground_truth_input,
            depth=num_classes)
        dict['_ground_truth_input_one_hot'] = _ground_truth_input_one_hot

        _learning_rate_input = tf.placeholder(
            # tf.float64,
            tf.float32,
            [],
            name='learning_rate_input')
        dict['_learning_rate_input'] = _learning_rate_input

        components = []
        for key in modalities:
            modality = tf.gather(
                params=_fingerprint_input,
                indices=modalities_assoc[config.POSITION][key],
                axis=2)

            batch_normalization = True

            # expand dimensions of `inputs`, i.e. (?, ?) -> (?, ?, 1)
            # _inputs = tf.expand_dims(inputs, 2)
            _inputs = modality
            for i, kernel_size in enumerate(kernel_sizes[key]):
                # conv1
                conv_1 = tf.layers.conv1d(
                    inputs=_inputs,
                    filters=num_filters,
                    kernel_size=int(kernel_size),
                    strides=int(kernel_size * (1 - overlap)),
                    padding='valid',
                    name=key + '_conv_' + str(i))
                dict[key + '_conv_' + str(i)] = conv_1

                bn_conv_1 = conv_1
                if batch_normalization is True:
                    bn_conv_1 = tf.layers.batch_normalization(conv_1, training=_training)

                relu_1 = tf.nn.relu(bn_conv_1)
                dict[key + '_relu_' + str(i)] = relu_1

                # max-pooling layer
                pool_1 = tf.layers.max_pooling1d(
                    inputs=relu_1,
                    # `pool_size` in other terms, how many features vectors to pull
                    # together while sliding over the sequence
                    pool_size=2,
                    # `strides` in this context means how many overlaping features
                    # vectors from window i will be merged with features vectors from
                    # window i+1
                    strides=2,
                    padding='valid',
                    name=key + '_pool_' + str(i))
                dict[key + '_pool_' + str(i)] = pool_1

                _inputs = pool_1  # for the next iteration

            components.append(_inputs)

        # concatenate learned features from each component of the modality
        concatenate_features = tf.concat(
            values=components,
            # axis=2,
            axis=1,
            name='concatenate_features')
        dict['concatenate_features'] = concatenate_features

        seq_lens = []
        for key in modalities:
            p3 = self._get_seq_len(
                initial_size=6000,
                kernel_sizes=kernel_sizes[key],
                overlap=overlap)

            seq_lens.append(p3)

        print('seq_lens = %s' % (seq_lens,))

        # [batch_size, seq_len_mod_1 + ... + seq_len_mod_n, num_filters * num_modalities]
        #                  -> [batch_size, (seq_len_mod_1 + ... + seq_len_mod_2) * num_filters * num_modalities]
        concatenate_features_flat = tf.reshape(
        #    concatenate_features, [-1, int(p3) * num_filters * num_modalities])
            concatenate_features, [-1, sum(seq_lens) * num_filters])
        dict['concatenate_features_flat'] = concatenate_features_flat

        dense = tf.layers.dense(
            inputs=concatenate_features_flat,
            units=num_units_dense_layer)
        dict['dense'] = dense

        relu = tf.nn.relu(dense)
        # relu = tf.nn.tanh(dense)
        dict['relu'] = relu

        dropout = tf.layers.dropout(
            inputs=relu,
            rate=dropout,
            training=_training)

        _logits = tf.layers.dense(
            inputs=dropout,
            units=num_classes)
        dict['_logits'] = _logits

        # loss_function
        with tf.name_scope('cross_entropy'):
            # cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
            _cross_entropy_mean = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    # labels=labels,
                    labels=_ground_truth_input_one_hot,
                    logits=_logits))
        dict['_cross_entropy_mean'] = _cross_entropy_mean

        # optimization
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            # update_ops in this case include updating moving_mean and moving_variance
            # of batch normalization operations.
            with tf.name_scope('train'):
                _train_step = tf.train.GradientDescentOptimizer(
                    _learning_rate_input).minimize(_cross_entropy_mean)
        dict['_train_step'] = _train_step

        # model evaluation
        _predicted_indices = tf.argmax(_logits, 1)
        _probs = tf.nn.softmax(_logits)
        dict['_predicted_indices'] = _predicted_indices
        dict['_probs'] = _probs

        _correct_prediction = tf.equal(
            _predicted_indices,
            _ground_truth_input)
        dict['_correct_prediction'] = _correct_prediction

        _confusion_matrix = tf.confusion_matrix(
            _ground_truth_input,
            _predicted_indices,
            num_classes=num_classes)
        dict['_confusion_matrix'] = _confusion_matrix

        _evaluation_step = tf.reduce_mean(
            tf.cast(_correct_prediction, tf.float64))
        dict['_evaluation_step'] = _evaluation_step

        _tfauc = tf.metrics.auc(
            _ground_truth_input_one_hot,
            _probs)
        dict['_tfauc'] = _tfauc

        # _session = tf.Session().as_default()
        with tf.Session().as_default() as _session:
            tf.global_variables_initializer().run()
        dict['_session'] = _session

        # The tf.train.Saver constructor adds save and restore ops to the graph
        # for all, or a specified list, of the variables in the graph.
        _saver = tf.train.Saver()
        dict['_saver'] = _saver

        # -------------------------------------------------
        nodes_to_evaluate = {
            '_ground_truth_input_one_hot': _ground_truth_input_one_hot,
            '_confusion_matrix': _confusion_matrix,
            '_predicted_indices': _predicted_indices,
            '_learning_rate_input': _learning_rate_input,
            '_cross_entropy_mean': _cross_entropy_mean,
            '_evaluation_step': _evaluation_step,
            '_train_step': _train_step,
            'concatenate_features': concatenate_features,
            'concatenate_features_flat': concatenate_features_flat,
            '_logits': _logits,
            'relu': relu,
            'dense': dense,
        }
        # modalities = self.assoc.keys()
        for key in modalities:
            for i in range(len(kernel_sizes[key])):
                conv_layer_name = key + '_conv_' + str(i)
                nodes_to_evaluate[conv_layer_name] = dict[conv_layer_name]
                pool_layer_name = key + '_pool_' + str(i)
                nodes_to_evaluate[pool_layer_name] = dict[pool_layer_name]
        # -------------------------------------------------
        dict['nodes_to_evaluate'] = nodes_to_evaluate

        # nodes_to_persist = evaluation_results.copy()
        nodes_to_persist = dict.fromkeys(nodes_to_evaluate)

        del nodes_to_persist['_ground_truth_input_one_hot']
        del nodes_to_persist['_confusion_matrix']
        del nodes_to_persist['_predicted_indices']
        del nodes_to_persist['_learning_rate_input']
        del nodes_to_persist['_cross_entropy_mean']
        del nodes_to_persist['_evaluation_step']
        del nodes_to_persist['_train_step']

        # dict['nodes_to_persist'] = nodes_to_persist
        dict['nodes_to_persist'] = {}  # nothing to persist for now

        return dict

    def _convolutional_layers(self, inputs, name, num_filters, kernel_sizes,
                              # kernel_size_1, kernel_size_2, kernel_size_3,
                              overlap, training, batch_normalization):

        # expand dimensions of `inputs`, i.e. (?, ?) -> (?, ?, 1)
        # _inputs = tf.expand_dims(inputs, 2)
        _inputs = inputs

        # convolutional layers
        # conv1
        conv_1 = tf.layers.conv1d(
            inputs=_inputs,
            filters=num_filters,
            kernel_size=int(kernel_sizes[0]),
            strides=int(kernel_sizes[0] * (1 - overlap)),
            padding='valid',
            # activation=tf.nn.relu,
            name=name + '_conv_1')

        bn_conv_1 = conv_1
        if batch_normalization is True:
            bn_conv_1 = tf.layers.batch_normalization(conv_1, training=training)

        relu_1 = tf.nn.relu(bn_conv_1)

        # max-pooling layer
        pool_1 = tf.layers.max_pooling1d(
            inputs=relu_1,
            # `pool_size` in other terms, how many features vectors to pull
            # together while sliding over the sequence
            pool_size=2,
            # `strides` in this context means how many overlaping features
            # vectors from window i will be merged with features vectors from
            # window i+1
            strides=2,
            padding='valid',
            name=name + '_pool_1')

        # conv2
        conv_2 = tf.layers.conv1d(
            inputs=pool_1,
            filters=num_filters,
            kernel_size=int(kernel_sizes[1]),
            strides=int(kernel_sizes[1] * (1 - overlap)),
            padding='valid',
            # activation=tf.nn.relu,
            name=name + '_conv_2')

        bn_conv_2 = conv_2
        if batch_normalization is True:
            bn_conv_2 = tf.layers.batch_normalization(conv_2, training=training)

        relu_2 = tf.nn.relu(bn_conv_2)

        # max-pooling layer 2
        pool_2 = tf.layers.max_pooling1d(
            inputs=relu_2,
            pool_size=2,
            strides=2,
            padding='valid',
            name=name + '_pool_2')

        # conv3
        conv_3 = tf.layers.conv1d(
            inputs=pool_2,
            filters=num_filters,
            kernel_size=int(kernel_sizes[2]),
            strides=int(kernel_sizes[2] * (1 - overlap)),
            padding='valid',
            # activation=tf.nn.relu,
            name=name + '_conv_3')

        bn_conv_3 = conv_3
        if batch_normalization is True:
            bn_conv_3 = tf.layers.batch_normalization(conv_3, training=training)

        relu_3 = tf.nn.relu(bn_conv_3)

        # max-pooling layer 3
        pool_3 = tf.layers.max_pooling1d(
            inputs=relu_3,
            pool_size=2,
            strides=2,
            padding='valid',
            name=name + '_pool_3')

        return pool_3

    def restore_model(self):
        """ Should have called __setstate__ before calling this function.
        """
        self._saver.restore(
            self._session,
            './predictive-model')


if __name__ == '__main__':
    # Test things out
    tf.logging.set_verbosity(tf.logging.INFO)
    fire.Fire(Estimator)
