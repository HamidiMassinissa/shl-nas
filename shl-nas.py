from config import Configuration


if __name__ == '__main__':

    config = Configuration()
    config.parse_commandline()
    config.new_experiment()

    if config.RUN == 'bayesopt':
        from bayesOpt import runBayes
        runBayes()

    if config.RUN == 'trainSingleModel':
        from testing import trainSingleModel
        trainSingleModel()
