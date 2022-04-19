import sqlalchemy as sqla
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import sys
# import for create table


Base = declarative_base()
import pandas as pd


class SQLLiteAccess:
    session = None

    def __init__(self, logger, showSQL=False):
        import Data.models
        import util.html
        self.logger = logger
        self.engine = sqla.create_engine(
            'sqlite:///Data/Data.db', echo=showSQL)
        self.meta = sqla.MetaData(bind=self.engine, reflect=True)
        Base.metadata.create_all(self.engine, checkfirst=True)
        self.conn = self.engine.connect()
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    # def create_tables(self):
        # self.Base.metadata.create_all(self.engine,checkfirst=True)

    def now():
        return datetime.datetime.now()

    def addRow(self, row_info):

        try:
            self.session.add(row_info)
            self.session.commit()
        except sqla.exc.OperationalError:
            self.session.rollback()
            raise

        except BaseException:
            self.session.rollback()
            self.logger.warning(
                'results not saved for {} due to {}'.format(
                    str(row_info), sys.exc_info()))


class PostgresAccess:
    session = None

    def __init__(self, logger, showSQL=False):
        import Data.models
        import util.html
        self.logger = logger

        baseUrl = 'postgresql://{}:{}@{}:{}/{}'
        url = baseUrl.format('postgres', 'root', 'localhost', '5432', 'Data')
        self.engine = sqla.create_engine(
            url, client_encoding='utf8', echo=showSQL)
        self.meta = sqla.MetaData(bind=self.engine, reflect=True)
        Base.metadata.create_all(self.engine, checkfirst=True)
        self.meta.create_all(self.engine, checkfirst=True)
        self.conn = self.engine.connect()
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def now():
        return datetime.datetime.now()

    def to_sql_k(self, sqlBuilder, frame, if_exists='fail', index=True,
                 index_label=None, schema=None, chunksize=None, dtype=None, **kwargs):

        if dtype is not None:
            from sqlalchemy.types import to_instance, TypeEngine
            for col, my_type in dtype.items():
                if not isinstance(to_instance(my_type), TypeEngine):
                    raise ValueError('The type of %s is not a SQLAlchemy '
                                     'type ' % col)

        table = pd.io.sql.SQLTable(frame.name, sqlBuilder, frame=frame, index=index,
                                   if_exists=if_exists, index_label=index_label,
                                   schema=schema, dtype=dtype, **kwargs)
        table.create()
        table.insert(chunksize)

    def dfToTable(self, df, if_exists='append', name=None, index=True,
                  index_label=None, schema=None, chunksize=None, dtype=None, **kwargs):
        pandas_sql = pd.io.sql.pandasSQL_builder(
            self.engine, schema=None, flavor=None)
        if name:
            df.name = name
        self.to_sql_k(
            pandas_sql,
            df,
            index=True,
            index_label=df.index.names,
            keys=df.index.names,
            if_exists=if_exists,
            schema=schema)

    def addRow(self, row_info, reraiseError=True):

        try:

            self.session.add(row_info)
            self.session.commit()

        except sqla.exc.IntegrityError:
            self.session.rollback()
            self.logger.warning(
                'results not saved for {} due to {}'.format(
                    str(row_info), sys.exc_info()))
            if reraiseError:
                raise

        except sqla.exc.OperationalError:
            self.session.rollback()
            self.logger.warning(
                'results not saved for {} due to {}'.format(
                    str(row_info), sys.exc_info()))
            if reraiseError:
                raise

        except BaseException:
            self.session.rollback()
            self.logger.warning(
                'results not saved for {} due to {}'.format(
                    str(row_info), sys.exc_info()))
            if reraiseError:
                raise
