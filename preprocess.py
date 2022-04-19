import Data.queries as queries
import Data.models as models
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
# -*- coding: utf-8 -*-

from util.text import isHumanized, Dehumanize
from util.memorize import Memorize


class Model_Data:
    def __init__(self, logger, DataAccess):
        self.logger = logger
        self.DA = DataAccess
        self.cat_cols = []
        self.target_cols = []
        self.num_cols = []
        self.ignore_cols = []
        self.id_cols = []
        self.unk_cols = []
        self.humanized_cols = []
        self.possible_feature_names = []
        self.le = LabelEncoder()

    def _dTypeToRole(self, dTypeName, colName):

        arr = {
            'float64': 'interval',
            'int64': 'interval',
            '<M8[ns]': 'nominal',
            'O': 'nominal',
            'object': 'nominal',
            "dtype('O')": 'nominal'
        }

        if colName == 'target':
            return 'target'
        if dTypeName in arr:
            return arr[dTypeName]
        else:
            self.logger.error('dType not found:{}'.format(dTypeName))
            return dTypeName

    def _getDTypeName(self, dType):
        nameArr = dType.name.split(".")
        dTypeName = nameArr[len(nameArr) - 1]
        return dTypeName

    def setInitialRoleAndProfile(self, modelName, fullData, columnName):

        col = fullData[columnName]
        dName = self._getDTypeName(col.dtype)
        rowCnt = len(fullData.index)
        suggested_role = self._dTypeToRole(dName, columnName)
        percnull = col.isnull().sum() / rowCnt
        val1 = str(col.iloc[0])[:250]
        val2 = str(col.iloc[1])[:250]
        humanized = (dName == 'object' and isHumanized(
            val1) and isHumanized(val2))
        if suggested_role == 'interval':
            percneg = (col < 0).sum() / rowCnt
        else:
            percneg = -1
        #self.logger.debug('setting initial role for {} isinf:{}  isNan:{}'.format(columnName,isinf,isnan))
        mdm = models.ModelDataMeta(model_name=modelName,
                                   column_name=columnName,
                                   dtype=dName,
                                   col_val1=val1,
                                   col_val2=val2,
                                   suggested_role=suggested_role,
                                   perc_null=percnull,
                                   perc_neg=percneg,
                                   humanized=humanized
                                   )

        self.DA.addRow(mdm)

    def getxy(self, seed, samplesize=0, weights=None):
        # Split-out validation dataset

        if samplesize == 0:
            sampledData = self.modelData
        else:
            sampledData = self.modelData.sample(
                samplesize, random_state=seed, weights=weights)

        self.x_train = sampledData[list(self.possible_feature_names)].values
        self.y_unraveled = sampledData[list(self.target_cols)].values

        self.y_raveled = self.y_unraveled.ravel()
        return self.x_train, self.y_raveled

    def categorizeColumn(self, mdm):
        role = mdm.role or mdm.suggested_role
        if role == 'target':
            self.target_cols.append(mdm.column_name)
        elif role == 'nominal':
            self.cat_cols.append(mdm.column_name)
        elif role == 'interval' or role == 'ordinal':
            self.num_cols.append(mdm.column_name)
        elif role == 'ignore':
            self.ignore_cols.append(mdm.column_name)
        elif role == 'id':
            self.id_cols.append(mdm.column_name)
        elif mdm.humanized == 'true':
            self.humanized_cols.append(mdm.column_name)
        else:
            self.unk_cols.append(mdm.column_name)
            self.logger.error('unknown role for {}'.format(mdm.column_name))

    def transformColumnWise(self, full_data, column_names, function):
        for column_name in column_names:
            column = full_data[column_name].to_frame()
            full_data[column_name] = column.apply(function)

    def transformElementWise(self, full_data, column_names, function):
        for column_name in column_names:
            column = full_data[column_name].to_frame()

            full_data[column_name] = column.applymap(function)

    def catTransform(self, column):

        column = self.le.fit_transform(column.astype('str'))
        return column

    def naTransform(self, column):
        isinf = np.isinf(column).any()
        isnan = np.isnan(column).any()
        if isnan or isinf:

            column.fillna(-1, inplace=True)
        return column

    # Summing up, apply works on a row / column basis of a DataFrame,
    # applymap works element-wise on a DataFrame,
    # and map works element-wise on a Series

    def transformAllColumns(self, fulldata, cat_cols,
                            num_cols, humanized_cols):
        self.transformColumnWise(fulldata, cat_cols, self.catTransform)
        self.transformColumnWise(fulldata, num_cols, self.naTransform)
        self.transformElementWise(fulldata, humanized_cols, Dehumanize)

        return fulldata
        #fullData[column] = number.fit_transform(fullData[column].astype('str'))

    def getMdm(self, modelName, fullData, column, isDirty):
        mdm = queries.checkIfModelDataMetaStored(self.DA, modelName, column)

        if not mdm:
            self.setInitialRoleAndProfile(modelName, fullData, column)
            isDirty = True
            mdm = queries.checkIfModelDataMetaStored(
                self.DA, modelName, column)
        return mdm, isDirty

    @Memorize
    def getProcessedData(self, modelName, sourceTableName):
        self.logger.reportElapsedTime('setting roles and profiling')
        le = LabelEncoder()
        isDirty = False

        fullData = pd.read_sql(
            'select * from {}'.format(sourceTableName),
            self.DA.engine)

        for column in fullData:

            mdm, isDirty = self.getMdm(modelName, fullData, column, isDirty)
            self.categorizeColumn(mdm)

        self.logger.reportElapsedTime('transformAllColumns')
        self.transformAllColumns(
            fullData,
            self.cat_cols,
            self.num_cols,
            self.humanized_cols)

        if isDirty:
            self.logger.info('new column added to roles')

        self.possible_feature_names = list(set(list(fullData.columns)) - set(
            self.id_cols) - set(self.target_cols) - set(self.ignore_cols) - set(self.unk_cols))
        self.isDirty = isDirty
        self.modelData = fullData
        return fullData
