import sqlalchemy as sqla
import .Data.DataAccess as da
import datetime
from datetime import date

class ModelResults(da.Base):
    __tablename__ = "model_meta"
    __table_args__ = ({"schema": "model_data"})
    model_name = sqla.Column(sqla.String(150), primary_key=True)
    run_type_cd = sqla.Column(sqla.Integer, primary_key=True)
    run_type = sqla.Column(sqla.String(150))
    pipeline = sqla.Column(sqla.String(150), primary_key=True)
    classifier = sqla.Column(sqla.String(150))
    usable_parameters = sqla.Column(sqla.UnicodeText())
    best_parameters = sqla.Column(sqla.UnicodeText())
    features = sqla.Column(sqla.UnicodeText())
    pickle = sqla.Column(sqla.PickleType)
    scoretype = sqla.Column(sqla.String(150))
    score = sqla.Column(sqla.Numeric)
    score_notes = sqla.Column(sqla.String(255))
    confusion_matrix = sqla.Column(sqla.UnicodeText())

    def __repr__(self):
        return '<{} {} Results with pipeline {}: {} {}>'.format(
            self.run_type, self.model_name, self.pipeline, self.score, self.score_notes)


class ModelDataMeta(da.Base):
    __tablename__ = "model_data_meta"
    __table_args__ = ({"schema": "model_data"})
    model_name = sqla.Column(sqla.String(150), primary_key=True)
    column_name = sqla.Column(sqla.String(150), primary_key=True)
    dtype = sqla.Column(sqla.String(50))
    col_val1 = sqla.Column(sqla.String(250))
    col_val2 = sqla.Column(sqla.String(255))
    suggested_role = sqla.Column(sqla.String(50))
    role = sqla.Column(sqla.String(50))
    perc_null = sqla.Column(sqla.Numeric)
    perc_neg = sqla.Column(sqla.Numeric)
    humanized = sqla.Column(sqla.Boolean)

    def __repr__(self):
        return '<metadata for {} and {} >'.format(
            self.model_name, self.column_name)