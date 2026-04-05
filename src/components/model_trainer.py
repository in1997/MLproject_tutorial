import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost
import catboost

from src.exception import CustomException
from src.logger import logging
import sys
import os
from dataclasses import dataclass

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split data into x and y features")
            x_train, y_train = train_array[:, :-1], train_array[:, -1]
            x_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(),
                "Random Forest Regressor": RandomForestRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "Support Vector Regressor": SVR(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGB Regressor": xgboost.XGBRegressor(),
                "CatBoost Regressor": catboost.CatBoostRegressor(verbose=False)
            }

            model_report: dict = evaluate_models(x_train, y_train, x_test, y_test, models)

            logging.info("model report generated")

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]
            logging.info(f"Best model found on both training and testing dataset is {best_model_name} with r2 score: {best_model_score}")  

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)
            
            logging.info("Saving the best model")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_score
    
        except Exception as e:
            raise CustomException(e, sys)