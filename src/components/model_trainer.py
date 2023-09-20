import os 
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","Model.pkl")

class ModelTrainer:
    def __init(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array,preprocessor_path):
        try:
            logging.info("Split training and test input data")
            X_train,Y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = (
                LinearRegression(),
                DecisionTreeRegressor(),
                RandomForestRegressor(),
                AdaBoostRegressor(),
                GradientBoostingRegressor(),
                XGBRegressor(),
                KNeighborsRegressor(),
                CatBoostRegressor(),
            )

            # model_report: dict = evaluate_model{X_train=X_train,Y_train=y_train,X_test,y_test,models=models}
            
           
        except:
            pass

