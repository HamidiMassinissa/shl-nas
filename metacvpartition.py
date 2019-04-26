import numpy as np
import math


class metacvpartition():

    def __init__(self, labels, nFolds, metaSegmentLength, debug=False):
        """
           Meta-segmented cross-validation [1].

           C = metacvpartition(labels, nFolds, metaSegmentLength);

           labels is a Nx1 matrix with (integer) labels. nFolds is the
           number of folds in the cross-validation. metaSegmentLength
           is the number of frames in each meta-segment.

           C is an object with a similar interface to cvpartition.
               C.training(i)   Nx1     Indicator-matrix for
                                       training-set i.
               C.test(i)       Nx1     Indicator-matrix for test-set
                                       i.
           Other fields:
               C.numtestSets
               C.foldDistribution
               C.classDistribution

           [1] Hammerla, Nils Y., and Thomas Plötz. "Let's (not) stick together:
               pairwise similarity biases cross-validation in activity
               recognition." Proceedings of the 2015 ACM international joint
               conference on pervasive and ubiquitous computing. ACM, 2015.

        """

        # initialize
        self.N = len(labels)
        self.numTestSets = nFolds

        # number of meta-segments
        nP = math.ceil(self.N / metaSegmentLength)
        if debug:
            print('nP =', nP)
        self.indices = np.zeros((self.N, 1))
        if debug:
            print('indices =', self.indices)

        # get classes
        c = np.unique(labels)
        if debug:
            print('c =', c)
        # transform to integer labels
        L = labels.astype(int)
        for i in c:
            L[labels == i] = i
        if debug:
            print('L =', L)

        # get overall distribution of labels
        self.classDistribution = np.bincount(L).T
        if debug:
            print('classDistribution =', self.classDistribution)

        # initialize met-segment class distribution matrix
        cDist = np.zeros((int(nP), len(c)))
        self.foldDistributions = np.zeros((nFolds, len(c)))

        # estimate class distributions for each meta-segment
        for i in range(int(nP)):
            # get meta-segment label-distribution
            if i*metaSegmentLength <= self.N:
                l = L[i*metaSegmentLength:(i+1)*metaSegmentLength]
                if debug:
                    print('i*metaSegmentLength = ', i*metaSegmentLength)
            else:
                l = L[i*metaSegmentLength:-1]
            print('l =', l)

            # get labels unique to this meta-segment
            d = np.unique(l)
            if debug:
                print('d =', d)

            # save in matrix
            dl = np.bincount(l)
            if debug:
                print('dl =', dl)
            dl[dl > 0] += np.add(
                dl[dl > 0 ]
                , np.random.random((sum(dl > 0),)) * 0.1
                , out=dl[dl > 0]
                , casting='unsafe'
            )
            # assign non-zero elements
            cDist[i, d] = dl[dl > 0]
            if debug:
                print('cDist =', cDist)
            # add some noise for randomness of xval
            cDist[i, :] = cDist[i, :]

        # Here comes the trick: sort lexicographically
        # [~, I] = np.lexsort(cDist)
        if debug:
            print('cDist.shape = ', cDist.shape)
        I = np.lexsort([col for col in cDist.T], axis=0)
        if debug:
            print('I = ', I)

        # "I" now contains sorted list of distributions (ascending)
        # Now: assign folds
        # ind = 1 + mod(1:len(I), nFolds)
        # ind = 1 + np.mod(np.arange(len(I)), nFolds)
        ind = np.mod(np.arange(len(I)), nFolds)
        if debug:
            print('ind = ', ind)
        ind[I] = ind
        if debug:
            print('ind[I] =', ind[I])

        # save fold-wise distibutions for reference
        for i in range(nFolds):
            d = np.sum(cDist[ind == i, :], axis=0)
            if debug:
                print('cDist[ind == i, :] = ', cDist[ind == i, :])
                print('d = ', d)
            self.foldDistributions[i, :] = d / np.sum(d)
        if debug:
            print('foldDistributions = ', self.foldDistributions)

        # assign fold to each sample
        for i in range(int(nP)):
            self.indices[i*metaSegmentLength:(i+1)*metaSegmentLength] = ind[i]
        if debug:
            print('indices = ', self.indices)

        # make sure the indices it's the right size
        # self.indices = self.indices[1:self.N]
        self.indices = self.indices[0:self.N].flatten().astype(int)

        self.TestSize = np.bincount(self.indices).T
        # self.TrainSize = size(self.indices, 1) - self.TestSize
        self.TrainSize = self.indices.shape[0] - self.TestSize

    # def trainIndices(self, cv, fold):
    def trainIndices(self, fold):
        # return binary training mask from fold `fold`
        return np.flatnonzero(self.indices != fold)

    # def testIndices(self, cv, fold):
    def testIndices(self, fold):
        # return binary testing mask from fold `fold`
        return np.flatnonzero(self.indices == fold)

    def splitsGenerator(self):
        for fold in range(self.numTestSets):
            yield(
                self.trainIndices(fold),
                self.testIndices(fold)
            )


if __name__ == '__main__':
    # test things out
    import dataset
    import pipeline

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

    mxval = metacvpartition(
        y[:, 0]
        , nFolds=5
        , metaSegmentLength=10
        , debug=True)
