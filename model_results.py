# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 06:27:41 2017

@author: j
"""

from sklearn.metrics import confusion_matrix
import Data.models as models
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.pipeline import Pipeline
from sklearn.grid_search import RandomizedSearchCV


class ModelResults:
    def __init__(self, logger, DA):
        self.logger = logger
        self.DA = DA

        self.results = []

    def getSupport(self, step):
        try:
            supp = step.get_support()
        except BaseException:
            supp = None
        return supp

    def get_feature_df(self, pipeline, possible_feature_names):

        reversedSteps = reversed(list(enumerate(pipeline.steps)))

        for i, step in reversedSteps:

            transform = step[1]
            support = self.getSupport(transform)

            if not support is None:
                selectedFeatures = []  # The list of your K best features
                for bool, feature in zip(support, possible_feature_names):
                    if bool:
                        selectedFeatures.append(feature)
                return str(selectedFeatures)
        return ''

    def panda_confuse(self, y_act, y_pred):
        import pandas as pd

        ys_actu = pd.Series(list(y_act), name='Actual')
        ys_pred = pd.Series(list(y_pred), name='Predicted')

        df_confusion = pd.crosstab(ys_actu, ys_pred)
        return df_confusion

    def getResults(self, steps, x, y, modelName, rts, scoring,
                   possible_parameters, prh, possible_feature_names):

        pipe = Pipeline(steps=steps)
        stepNames = ','.join([s[0] for s in pipe.steps])
        classifier = pipe.steps[len(pipe.steps) - 1][0]

        if len(prh.top_classifiers) > 0 and classifier not in prh.top_classifiers:
            #self.logger.warning('{} not run as it is not a top classifier {}'.format(classifier,prh.top_classifiers))
            return

        if len(prh.top_pipelines) > 0 and stepNames not in prh.top_pipelines:
            self.logger.warning(
                '{} not run as it is not a top pipeline'.format(stepNames))
            return

        usableParams = prh.getUsableParams(
            possible_parameters, stepNames, steps, rts)

        note = ''
        selectedFeatures = ''
        confuse = ''
        result = -1
        params = str(usableParams)

        alreadyRan = prh.isAlreadyRan(modelName, rts.order, stepNames)
        if alreadyRan:
            return
        try:
            maxNumParam = max(rts.numberOfParameters, 1)
            rs = RandomizedSearchCV(
                pipe,
                usableParams,
                scoring=scoring,
                cv=rts.n_splits,
                n_iter=maxNumParam)
            rs.fit(x, y)
            result = rs.best_score_
            pipe = rs.best_estimator_
            y_pred = pipe.predict(x)
            params = str(rs.best_params_)

            selectedFeatures = self.get_feature_df(
                pipe, possible_feature_names)
            confuse = self.panda_confuse(y, y_pred)
            if rts.isFinal():
                
                self.logger.debug('outputting final')
                frames = pd.DataFrame(np.concatenate((x,y),axis=1))
                frames["y_pred"] = y_pred
                frames.to_csv(r"C:\Users\j\Google Drive\Projects\Iris\best_model_data.csv", sep=',')
                

            #print(confusion_matrix(y, y_pred))

        except ValueError as e:
            note = str(e)
            result = -1
            params = str(usableParams)

        except Exception as e:
            note = str(e)
            result = -1
            params = str(usableParams)

            raise

        finally:

            mr = models.ModelResults(
                model_name=modelName,
                run_type_cd=rts.order,
                run_type=rts.name,
                pipeline=stepNames,
                features=selectedFeatures,
                classifier=classifier,
                confusion_matrix=str(confuse),
                # pickle = pipe,
                scoretype=scoring,
                usable_parameters=str(usableParams),
                best_parameters=params,
                score=round(result, 3),
                score_notes=note)
            self.DA.addRow(mr)
            self.results.append(mr)

            self.logger.debug('running {}'.format(stepNames))
