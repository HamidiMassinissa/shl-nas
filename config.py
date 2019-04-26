import os
import datetime
from argparse import ArgumentParser


class Configuration:

    # experiment version
    MAJOR_VERSION = '5'
    MINOR_VERSION = '5'
    VERSION = MAJOR_VERSION + '.' + MINOR_VERSION
    REVISION = '3'
    MINOR_REVISION = '7'

    # mastering randomness
    SEED = 1

    # bayesian optimization meta-parameters
    N_CALLS = 60
    N_JOBS_bayes = -1

    # Hold-out ('ho') vs. Cross-validation ('cv')
    # VALIDATION = 'ho'
    VALIDATION = 'cv'
    # VALIDATION = 'pitfalls'

    # cross validation
    N_JOBS_cv = 1

    # mock data or load real data
    MOCK = False

    # debug
    DEBUG = True

    RUN = ''

    # where generated files, e.g. .mmap, slurm*, *.sav files are stored
    experimentsfolder = os.path.join('generated', VERSION)
    persistencefolder = os.path.join(experimentsfolder, REVISION)
    experiment_persistence = experimentsfolder
    bo_run_persistence = experiment_persistence

    # where extracted data live
    datafolder = os.path.join('generated', 'tmp')

    num_classes = 9
    batch_size = 50

    max_training_epochs = 1 # 52
    min_training_epochs = 12
    max_subsequent_epochs = 8

    # Shl-dataset related constants
    USER = ''
    DAY = ''
    POSITION = ''

    # hyperparameters for testing individual instantiations
    learning_rate = 0
    num_filters = 0
    kernel_sizes_list = []
    overlap = 0
    num_units_dense_layer = 0
    dropout = 0

    # validation parameters
    user_out = ''
    day_out = ''

    # metasegmented cross-validation parameters
    xval_nfolds = 5
    xval_metasegmentlength = 10

    # classes that can be experimented when filtering one-day out
    odo_classes = []

    @classmethod
    def parse_commandline(self, is_testing=False):
        """
        Synopsis
         Alter class attributes defined above with command line arguments
        """

        parser = ArgumentParser(description='')

        parser.add_argument(
            '--run',
            metavar='run',
            default='bayesopt',
            required=True
        )

        parser.add_argument(
            '--minor-revision',
            metavar='minor_revision',
            default='0',
            required=False
        )

        parser.add_argument(
            '--validation'
            , metavar='validation'
            , choices=(
                'ho'
                , 'cv'
                , 'pitfalls'
                , 'one-day-out'
                , 'one-user-out'
                , 'meta-segmented-cv'
            )
            , default='meta-segmented-cv'
            , required=False
        )

        parser.add_argument(
            '--num-classes'
            , metavar='num_classes'
            , type=int
            , default=9
            , required=False
        )

        parser.add_argument(
            '--batch-size'
            , metavar='batch_size'
            , type=int
            , default=50
            , required=False
        )

        parser.add_argument(
            '--user'
            , metavar='user'
            , choices=(
                'User1'
                , 'User2'
                , 'User3'
            )
            , required=False
        )

        parser.add_argument(
            '--day'
            , metavar='day'
            , required=False
        )

        parser.add_argument(
            '--position'
            , metavar='position'
            , choices=(
                'Bag'
                , 'Hand'
                , 'Hips'
                , 'Torso'
            )
            , default='Hips'
            , required=False
        )

        # Hyperparamets instantiation by hand
        parser.add_argument(
            '--learning_rate'
            , metavar='learning_rate'
            , type=float
            , default=0.1
            # , required=is_testing
            , required=False
        )

        parser.add_argument(
            '--num_filters'
            , metavar='num_filters'
            , type=int
            , default=28
            # , required=is_testing
            , required=False
        )

        parser.add_argument(
            '--kernel_sizes_list'
            , metavar='kernel_sizes_list'
            , nargs='+'
            , type=int
            , default=[15,9,9, 9,15,9, 13,15,9, 10,14,12, 15,9,10, 9,9,12, 15,15,9]
            # , required=is_testing
            , required=False
        )

        parser.add_argument(
            '--overlap'
            , metavar='overlap'
            , type=float
            , default=0.6
            # , required=is_testing
            , required=False
        )

        parser.add_argument(
            '--num_units_dense_layer'
            , metavar='num_units_dense_layer'
            , type=int
            , default=2048
            # , required=is_testing
            , required=False
        )

        parser.add_argument(
            '--dropout'
            , metavar='dropout'
            , type=float
            , default=0.5
            # , required=is_testing
            , required=False
        )

        parser.add_argument(
            '--user-out'
            , metavar='user_out'
            , choices=(
                'User1'
                , 'User2'
                , 'User3'
            )
            , required=False
        )

        parser.add_argument(
            '--day-out'
            , metavar='day_out'
            , choices=(
                # User1
                '260617'
                , '220617'
                , '270617'

                # User2
                , '140717'
                , '180717'
                , '140617'

                # User3
                , '070717'
                , '030717'
                , '140617'
            )
            , required=False
        )

        parser.add_argument(
            '--xval-nfolds'
            , metavar='xval_nfolds'
            , type=int
            , default=5
            , required=False
        )

        parser.add_argument(
            '--xval-metasegmentlength'
            , metavar='xval_metasegmentlength'
            , type=int
            , default=10
            , required=False
        )

        parser.add_argument(  # odo: one-day-out
            '--odo-classes'
            , metavar='odo_classes'
            , nargs='+'
            , type=int
            , required=False
        )

        args = parser.parse_args()
        print('Args = %', args)

        self.RUN = args.run
        self.MINOR_REVISION = args.minor_revision
        self.VALIDATION = args.validation
        self.num_classes = args.num_classes
        self.batch_size = args.batch_size
        self.USER = args.user
        self.DAY = args.day
        self.POSITION = args.position

        # hyper-parameters
        self.learning_rate = args.learning_rate
        self.num_filters = args.num_filters
        self.kernel_sizes_list = args.kernel_sizes_list
        self.overlap = args.overlap
        self.num_units_dense_layer = args.num_units_dense_layer
        self.dropout = args.dropout

        # validation parameters
        self.user_out = args.user_out
        self.day_out = args.day_out

        # metasegmented cross-validation parameters
        self.xval_nfolds = args.xval_nfolds
        self.xval_metasegmentlength = args.xval_metasegmentlength

        self.odo_classes = args.odo_classes

        self.cmd_args = args

    @classmethod
    def __str__(cls):
        return ', '.join(
            '{}: {}\n'.format(k, v)
            for (k, v) in cls.__dict__.items()  # if k.startswith('_')
        )

    @classmethod
    def new_experiment(self):
        self.experiment_persistence = os.path.join(
            self.experiment_persistence, '{}'.format(datetime.datetime.now()))
        self.bo_run_persistence = self.experiment_persistence
        assert not os.path.exists(self.experiment_persistence)
        os.makedirs(self.experiment_persistence)
        print('results of this experiment can be found in %s', self.experiment_persistence)

    @classmethod
    def new_BO_run(self):
        self.bo_run_persistence = os.path.join(
            self.experiment_persistence, '{}'.format(datetime.datetime.now()))
        assert not os.path.exists(self.bo_run_persistence)
        os.makedirs(self.bo_run_persistence)
