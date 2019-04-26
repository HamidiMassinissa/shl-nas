"""
Synopsis
--------

 Highly unbalanced datasets are commonly encountered in real applications.
 Moreover, this is often accompanied with a poorly represented class of
 interest (we talk about degrees of imbalance that are extremely low such
 as 2.5% or even 1%). In such cases, The F-measure is the most used
 performance metric in order to validate how well classifiers behave.
 However, there are many subtle differences used to compute this metric,
 and this paper [1] study the impact of such differences. Here I propose
 a class that computes this metric according to the different methods
 enumerated in [1].

 [1] Forman, George, and Martin Scholz. "Apples-to-apples in cross-
 validation studies: pitfalls in classifier performance measurement."
 ACM SIGKDD Explorations Newsletter 12.1 (2010): 49-57.

Technicalities
--------------

In addition, methods implemented inside the class Fscore are also provided as
stand-alone functions for convinence. Here are some notes about their signature:

 (i) The functions defined hereafter, follow the signature required by
     `sklearn.metrics.make_scorer`, that is:
         score_func(y, y_pred, **kwargs)
     An example of a `score_func` is given here:
     http://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html

 (ii) Including additional parameters to the scoring function is performed
     through the `**kwargs` and have to be provided when making the call to
     `make_scorer`. Here is an example:
         make_scorer(sklearn.metrics.fbeta_score, beta=2)
         -- where beta is one of the keyworded argumens (`**kwargs`)
"""
import numpy as np
from sklearn.metrics import confusion_matrix


class Fscore(object):
    def __init__(self, y, y_pred, datasplits, debug=False):
        """
        Synopsis
         For each fold and each class, compute TP, TN, FP, and FN.

        """
        self.debug = debug

        self.nFolds = len(datasplits)
        self.nLabels = len(np.unique(y))

        TP = np.empty((self.nFolds, self.nLabels))
        TN = np.empty_like(TP)
        FP = np.empty_like(TP)
        FN = np.empty_like(TP)

        for fold, (_, test_index) in enumerate(datasplits):
            y_fold, y_pred_fold = y[test_index], y_pred[test_index]
            confmat_fold = confusion_matrix(y_fold, y_pred_fold,
                                            # ensure that all classes in `y` are
                                            # considered with the following:
                                            labels=np.unique(y))

            FP[fold, :] = confmat_fold.sum(axis=0) - np.diag(confmat_fold)
            FN[fold, :] = confmat_fold.sum(axis=1) - np.diag(confmat_fold)
            TP[fold, :] = np.diag(confmat_fold)
            TN[fold, :] = confmat_fold.sum() - (FP[fold, :] + FN[fold, :] + TP[fold, :])

            if self.debug is True:
                print('FP[fold, :] = %s' % (FP[fold, :],))
                print('FN[fold, :] = %s' % (FN[fold, :],))
                print('TP[fold, :] = %s' % (TP[fold, :],))
                print('TN[fold, :] = %s' % (TN[fold, :],))

        self.TP = TP.sum(axis=1)
        self.TN = TN.sum(axis=1)
        self.FP = FP.sum(axis=1)
        self.FN = FN.sum(axis=1)

    def precision(self):
        """
        Synopsis
         Compute precision for each fold.

         Returns
        """
        prec = self.TP / (self.TP + self.FP)

        if self.debug is True:
            print('precision.shape = %s' % (prec.shape,))

        return prec

    def recall(self):
        """
        Synopsis

         Returns
        """
        recall = np.divide(self.TP, (self.TP + self.FN))

        if self.debug is True:
            print('recall.shape = %s' % (recall.shape,))

        return recall

    def fmeasure(self, precisionArray, recallArray):
        """
        Synopsis

         Returns
        """
        fmeasure = 2 * np.divide(precisionArray * recallArray, precisionArray + recallArray)

        if self.debug is True:
            print('fmeasure.shape = %s' % (fmeasure.shape,))

        return fmeasure

    def precision_avg(self):
        """
        Synopsis

         Returns
        """
        prec = np.divide(self.TP, (self.TP + self.FP))
        prec_avg = 1 / self.nFolds * prec.sum(axis=1)

        return prec_avg

    def recall_avg(self):
        """
        Synopsis

         Returns
        """
        recall = np.divide(self.TP, (self.TP + self.FN));
        recall_avg = 1 / self.nFolds * recall.sum(axis=1);

        return recall_avg

    def weightedF1score(self, **kwargs):
        """
            As defined in ??? and used in the context of human activity recognition
            by [Hammerla et al. 2017].

            ---
            [hammerla et al. 2017] Nils Y. Hammerla, Shane Halloran, and Thomas
            Plötz. 2016. Deep, convolutional, and recurrent models for human
            activity recognition using wearables. In Proceedings of the Twenty-Fifth
            International Joint Conference on Artificial Intelligence (IJCAI'16),
            Gerhard Brewka (Ed.). AAAI Press 1533-1540.
        """
        pass

    def avg_fscore(self, **kwargs):
        """
        Synopsis
         for each fold, we compute the F-measure F^(i). The final estimate is the
         mean of all folds

         Returns
        """
        precisionArray = self.precision()
        recallArray = self.recall()

        if self.debug is True:
            print('precisionArray = %s' % (precisionArray,))
            print('recallArray = %s' % (recallArray,))

        fmeasureArray = self.fmeasure(precisionArray, recallArray)

        if self.debug is True:
            print('fmeasureArray = %s' % (fmeasureArray,))

        favg = np.mean(fmeasureArray)

        return favg

    def prre_fscore(self, **kwargs):
        """
        Synopsis

         Returns
        """
        precisionArray = self.precision()
        recallArray = self.recall()

        if self.debug is True:
            print('precisionArray = %s' % (precisionArray,))
            print('recallArray = %s' % (recallArray,))

        precision = 1 / self.nFolds * precisionArray.sum(axis=0)
        recall = 1 / self.nFolds * recallArray.sum(axis=0)

        if self.debug is True:
            print('precision = %s' % (precision,))
            print('recall = %s' % (recall,))

        fprre = 2 * np.divide(precision * recall, precision + recall)

        return fprre

    def tpfp_fscore(self, **kwargs):
        """
        Synopsis

         Returns
        """
        truePositive = self.TP.sum(axis=0)
        falsePositive = self.FP.sum(axis=0)
        falseNegative = self.FN.sum(axis=0)

        if self.debug is True:
            print('truePositive = %s' % (truePositive,))
            print('falsePositive = %s' % (falsePositive,))
            print('falseNegative = %s' % (falseNegative,))
            print('self.FN = %s' % (self.FN,))

        ftpfp = 2 * np.divide(truePositive, 2 * truePositive + falsePositive + falseNegative)
        toto = 2 * truePositive / (2 * truePositive + falsePositive + falseNegative)

        if self.debug is True:
            print('ftpfp = %s' % (ftpfp,))
            print('toto = %s' % (toto,))

        return ftpfp


if __name__ == '__main__':

    import dataset
    import pipeline
    from sklearn.model_selection import StratifiedKFold

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

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=1)

    fscore = Fscore(
        y=y[:, 0],
        y_pred=y[:, 0],
        datasplits=list(cv.split(X, y[:, 0])),
        debug=True)

    print(fscore.avg_fscore())
    print(fscore.prre_fscore())
    print(fscore.tpfp_fscore())

    assert fscore.avg_fscore() == 1.0
    assert fscore.prre_fscore() == 1.0
    assert fscore.tpfp_fscore() == 1.0
