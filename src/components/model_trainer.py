import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info('Splitting training and testing input data')
            X_train, y_train, X_test, y_test = (train_arr[:, :-1], train_arr[:, -1], test_arr[:, :-1], test_arr[:, -1])
            
            models = {
                'Adaboost Classifier': AdaBoostClassifier(),
                'Gradient Boosting Regressor': GradientBoostingRegressor(),
                'Random forest regressor': RandomForestRegressor(),
                'Linear Regression': LinearRegression(),
                'KNN': KNeighborsRegressor(),
                'Decision tree': DecisionTreeRegressor(),
                'Catboost': CatBoostRegressor(),
                'Xgboost': XGBRegressor()
            }
            
            model_report:dict=evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException('No best model found')
            
            logging.info('Best model found on both the trainin g and testing data')
            
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            
            predicted = best_model.predict(X_test)
            
            r2score = r2_score(y_test, predicted)
            
            return r2score
        
        except Exception as e:
            raise CustomException(e, sys)