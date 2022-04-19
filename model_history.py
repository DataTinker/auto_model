# -*- coding: utf-8 -*-
import pandas as pd
from Data import queries
import ast


class PriorRunHistory:
    def __init__(self, logger, alreadyRun, top_classes,
                 top_pipelines, top_params, pipes_already_ran):
        self.logger = logger
        self.alreadyRan = alreadyRun
        self.top_classifiers = top_classes
        self.top_pipelines = top_pipelines
        self.top_params = top_params
        self.pipes_already_ran = pipes_already_ran

    def isAlreadyRan(self, model_name, run_type_cd, pipeline):

        key = [model_name, run_type_cd, pipeline]

        return key in self.pipes_already_ran

    def isTopPipe(self, classifier, stepNames):
        if len(self.top_classifiers) > 0 and classifier not in self.top_classifiers:
            #self.logger.warning('{} not run as it is not a top classifier {}'.format(classifier,top_classifiers))
            return False

        if len(self.top_pipelines) > 0 and stepNames not in self.top_pipelines:

            return False
        return True

    def getUsableParams(self, possible_parameters, stepName, steps, rts):
        usableParams = {}

        if rts.numberOfParameters == 0:
            return {}
        elif rts.numberOfParameters == 1:
            listParams = {}
            topParam = self.top_params[stepName]
            for key, value in topParam.items():
                valueList = [value]

                listParams[key] = valueList
            return listParams
        else:
            #self.logger.debug('possible param keys : ' .format(pipe.get_params().keys()))

            for paramName, paramValue in possible_parameters.items():
                for step in steps:
                    stepName = step[0]
                    if stepName in paramName:
                        #    'SelectKBest__k':  range(5,10),
                        usableParams[paramName] = paramValue
            #self.logger.debug('usable parameters:{}'.format(usableParams))
            return usableParams


class ModelHistory:
    def __init__(self, logger, DataAccess, seed=7):
        self.logger = logger
        self.DA = DataAccess

    def getLargestList(self, df, groupField, valueField, n):
        maxDF = df.groupby(groupField, as_index=False).max()
        largest = maxDF.nlargest(n, valueField)
        return largest[groupField].tolist()

    def filterDF(self, df, filterName, filterValue):
        filtered = df[df[filterName] == filterValue]
        return filtered

    def processPriorRuns(self, modelName, rts, possibleSteps):
        self.logger.debug('checking for prior run')
        psCnts = len(possibleSteps)
        qry = queries.getPriorModelRunQry(self.DA, modelName, rts.order)

        # return this
        alreadyRan = False
        top_classifiers = []
        pipes_already_ran = []
        top_pipes = []
        top_params = {}

        if not qry.first():
            self.logger.debug('no prior run history')
        else:
            self.logger.debug('querying history into DF')
            df = pd.read_sql(qry.statement, self.DA.engine)

            runTypeField = 'run_type_cd'
            rtCnts = df.groupby(runTypeField).size().to_dict()

            for index, r in df.iterrows():

                key = [r['model_name'], r['run_type_cd'], r['pipeline']]
                pipes_already_ran.append(key)

            try:
                cCnt = rtCnts[rts.order]
            except KeyError:
                cCnt = 0

            if cCnt == psCnts:
                alreadyRan = True
            else:
                print('{}_{}'.format(cCnt, psCnts))
                alreadyRan = False

            priorRunType = rts.order - 1

            if priorRunType >= 1:

                priorRunDF = self.filterDF(df, runTypeField, priorRunType)
                classifierfieldName = 'classifier'
                pipeFieldName = 'pipeline'
                paramFieldName = 'best_parameters'
                scoreFieldName = 'score'
                top_classifiers = self.getLargestList(
                    priorRunDF, classifierfieldName, scoreFieldName, rts.numberOfClassifiers)

                if rts.numberOfPipes > 0:
                    for clsName in top_classifiers:
                        clsDF = self.filterDF(
                            priorRunDF, classifierfieldName, clsName)
                        class_top_pipes = self.getLargestList(
                            clsDF, pipeFieldName, scoreFieldName, rts.numberOfPipes)
                        top_pipes = top_pipes + class_top_pipes
                        for pipeName in class_top_pipes:
                            pipeDF = self.filterDF(
                                clsDF, pipeFieldName, pipeName)
                            pipe_top_params = self.getLargestList(
                                pipeDF, paramFieldName, scoreFieldName, 1)
                            top_params[pipeName] = ast.literal_eval(
                                pipe_top_params[0])

        self.logger.debug(
            'finished processing prior runs result alreadyran:{}'.format(alreadyRan))
        prh = PriorRunHistory(
            self.logger,
            alreadyRan,
            top_classifiers,
            top_pipes,
            top_params,
            pipes_already_ran)
        return prh
