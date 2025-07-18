import os
import sys
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

import numpy as np
import pandas as pd

from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e :
        raise CustomException(e,sys)
def evaluate_models(x_train, y_train, x_test, y_test, models, param):
    try:
        report = {}
        
        for model_name, model in models.items():
            print(f"\nEvaluating model: {model_name}")
            
            # Get parameter grid if available, else use empty dict
            para = param.get(model_name, {})

            if para:
                gs = GridSearchCV(model, para, cv=3, n_jobs=-1, verbose=0)
                gs.fit(x_train, y_train)
                best_model = gs.best_estimator_
            else:
                model.fit(x_train, y_train)
                best_model = model

            y_train_pred = best_model.predict(x_train)
            y_test_pred = best_model.predict(x_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
