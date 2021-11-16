# imports
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        '''returns a pipelined model'''
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        self.pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])

    # def run(self):
    #     """set and train the pipeline"""
    #     pass

    def run(self):
        '''returns a trained pipelined model'''
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)


    # def evaluate(self, X_test, y_test):
    #     """evaluates the pipeline on df_test and return the RMSE"""
    #     pass

    def evaluate(self, X_test, y_test):
        '''returns the value of the RMSE'''
        y_pred = self.pipeline.predict(X_test)
        self.rmse = compute_rmse(y_pred, y_test)

if __name__ == "__main__":
    # get data
    df_raw = get_data(nrows=100)

    # clean data
    df = clean_data(df_raw, test=False)

    # set X and y
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)

    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

    # train
    trainer = Trainer(X=X_train, y=y_train)
    trainer.run()

    # train(X_train, y_train, pipeline)

    # evaluate
    trainer.evaluate(X_test, y_test)
    print(f"rmse is {trainer.rmse}")

    # evaluate()

    # print('TODO')
