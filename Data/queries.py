from datetime import timedelta, date, datetime
from sqlalchemy.sql import func, text
from sqlalchemy.sql.expression import bindparam
from sqlalchemy import Interval
import sqlalchemy as sqla
from Data.models import   ModelDataMeta, ModelResults
from sqlalchemy.sql.functions import coalesce


def getPriorModelRunQry(dataAccess, model_name, run_type_cd):

    qry = dataAccess.session.query(ModelResults).filter(
        ModelResults.run_type_cd <= run_type_cd)

    return qry


def checkIfModelDataMetaStored(dataAccess, model_name, col_name):
    qry = dataAccess.session.query(ModelDataMeta).filter(
        ModelDataMeta.model_name == model_name,
        ModelDataMeta.column_name == col_name)
    if qry.first():
        return qry.first()
    return


def updateBaseModeData(DA):
    with DA.engine.connect() as con:
        con.execute(
            'insert into model_data.basemodeldata select * from model_Data.vw_nextmodeldata')
