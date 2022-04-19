import logging
from util.logging import SuperLogger
from enum import Enum
import itertools
import pandas as pd
import Data.DataAccess as DA
# Initialize
from sklearn.feature_selection import SelectKBest, SelectFpr, SelectPercentile
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, SGDClassifier, RidgeClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

import blaze
import preprocess

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from model_results import ModelResults
from model_history import ModelHistory


class RunType(Enum):
    PipelineAssessment = 1
    ParameterOptimization = 2
    StabilityCheck = 3
    FinalizeKFold = 4

class RunTypeSettings():
    def __init__(self, runtype):
        self.rt = runtype
        if runtype == RunType.PipelineAssessment:
            self.setSettings(
                n_splits=3,
                nClass=0,
                rowCap=0,
                nPipe=0,
                nParam=0,
                rt=runtype)

        elif runtype == RunType.ParameterOptimization:
            self.setSettings(
                n_splits=3,
                nClass=3,
                rowCap=0,
                nPipe=0,
                nParam=3,
                rt=runtype)

        elif runtype == RunType.StabilityCheck:
            # self.setSettings(n_splits=3,nClass=3,rowCap=0,nPipe=3,nParam=10,rt=runtype)
            self.setSettings(
                n_splits=3,
                nClass=3,
                rowCap=0,
                nPipe=3,
                nParam=1,
                rt=runtype)

        elif runtype == RunType.FinalizeKFold:
            # self.setSettings(n_splits=3,nClass=3,rowCap=0,nPipe=3,nParam=10,rt=runtype)
            self.setSettings(
                n_splits=10,
                nClass=3,
                rowCap=0,
                nPipe=1,
                nParam=1,
                rt=runtype)

    def isFinal(self):
        return self.rt == RunType.FinalizeKFold

    def setSettings(self, n_splits, nClass, rowCap, nPipe, nParam, rt):
        self.n_splits = n_splits
        self.numberOfClassifiers = nClass
        self.numberOfPipes = nPipe
        self.rowCap = rowCap
        self.numberOfParameters = nParam
        self.name = rt.name
        self.order = rt.value


class automodel:
    def __init__(self, logger, DataAccess, seed=7):
        self.logger = logger
        self.DA = DataAccess
        self.preprocessed = False
        self.seed = seed
        self.mp = preprocess.Model_Data(logger, DataAccess)
        self.mr = ModelResults(logger, DataAccess)

        self.scalerModels = (
            ('StandardScaler', StandardScaler()),
            ('RobustScaler', RobustScaler()),
            ('MixMaxScaler', MinMaxScaler())
        )

        self.featureSelectionModels = (
            ('SelectKBest', SelectKBest()),
            ('SelectFpr', SelectFpr()),
            ('SelectPercentile', SelectPercentile())
        )

        self.classificationModels = (
            ("RidgeClassifier", RidgeClassifier()),
            ("Perceptron", Perceptron()),
            ("kNN", KNeighborsClassifier()),
            ("Random forest", RandomForestClassifier()),
            ("SVC-L", LinearSVC()),
            ('SGD', SGDClassifier()),
            ('NB1', MultinomialNB()),
            ('NB2', BernoulliNB()))
           
        self.params = {
            'SelectKBest__k': range(5, 10),
            'SelectPercentile__percentile': range(5, 10),
            'SelectFpr__alpha': [.025, .05, .075, .1, .15, .2]
        }

    def getxy(self, mp, rows):

        x, y = mp.getxy(self.seed, samplesize=rows, weights=None)

        assert len(x) == len(y)
        return x, y

    def runAll(self, data, modelName, scoring='roc_auc'):
        for rt in RunType:
            self.runOne(data, modelName, rt, scoring)

    def runOne(self, data, modelName,
               runtype=RunType.PipelineAssessment, scoring='roc_auc'):
        self.logger.debug(
            'starting {} run for {}'.format(
                runtype.name, modelName))
        if self.preprocessed == False:
            fullData = self.mp.getProcessedData(modelName, data)
            self.preprocessed = True

        rts = RunTypeSettings(runtype)

        possibleSteps = list(
            itertools.product(
                self.scalerModels,
                self.featureSelectionModels,
                self.classificationModels))
        self.logger.reportElapsedTime('ModelHistory')
        mh = ModelHistory(self.logger, DataAccess)

        self.logger.reportElapsedTime('processPriorRuns')
        prh = mh.processPriorRuns(modelName, rts, possibleSteps)
        if prh is not None and prh.alreadyRan:
            self.logger.warning('already ran: exiting')
            return

        if self.mp.isDirty:
            self.logger.warning('EXITING: is dirty, check roles and rerun')
            exit()

        self.logger.debug('get xy')
        x, y = self.getxy(self.mp, rts.rowCap)

        for i, steps in enumerate(possibleSteps):
            self.logger.reportElapsedTime('getResults', 30)
            self.mr.getResults(
                steps,
                x,
                y,
                modelName,
                rts,
                scoring,
                self.params,
                prh,
                self.mp.possible_feature_names)

        for x in self.mr.results:
            self.logger.info(x)
        self.logger.reportElapsedTime('finished')

if __name__ == '__main__':

    logger = SuperLogger(logging.DEBUG)
    DataAccess = DA.PostgresAccess(logger)

    logger.info('got data')
    #finalData.fillna(value = -9999)
    houseData = pd.read_csv(r'E:\Drive\python\kaggle\HousePrices\train.csv')
    am = automodel(logger, DataAccess)
    am.runAll(houseData,'HousePricesKaggle')
    #am.runAll('model_data.finalmodeldata', 'losers', scoring='precision')
    #am.runOne(finalData, 'losers',RunType.ParameterOptimization,scoring='precision')
