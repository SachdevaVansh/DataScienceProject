import yaml 
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import os, sys 
import dill
import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


def read_yaml_file(file_path: str)->dict:
    try:
        with open(file_path, 'rb') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise NetworkSecurityException(e) from e

def write_yaml_file(file_path:str, content:object, replace:bool=False)->None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
            dir_name=os.path.dirname(file_path)
            os.makedirs(dir_name,exist_ok=True)
        with open(file_path,'w') as f:
            yaml.dump(content,f)

    except Exception as e:
        raise NetworkSecurityException(e) from e

def save_numpy_array_data(file_path: str, array:np.array)->None:
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,'wb') as f:
            np.save(f,array)
    except Exception as e:
        raise NetworkSecurityException(e) from e

def save_object(file_path:str , obj:object)->None:
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path,'wb') as f:
            pickle.dump(obj,f)
        logging.info("Object has been saved successfully")
    except Exception as e:
        raise NetworkSecurityException(e) from e

def load_object(file_path: str)->object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exist")
        with open(file_path,'rb') as f:
            return pickle.load(f)

    except Exception as e:
        raise NetworkSecurityException(e) from e

def load_numpy_array_data(file_path:str)->np.array:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exist")

        with open(file_path,'rb') as f:
            return np.load(f)
    except Exception as e:
        raise NetworkSecurityException(e) from e

def evaluate_models(X_train,y_train,X_test,y_test,models,params):
    try:
        report={}

        for i in range(len(list(models))):
            model=list(models.values())[i]
            para=params[list(models.keys())[i]]

            gs=GridSearchCV(model, para, cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred=model.predict(X_train)

            y_test_pred=model.predict(X_test)

            train_model_score= r2_score(y_train,y_train_pred)
            test_model_score= r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]]=test_model_score
        return report 
    except Exception as e:
        raise NetworkSecurityException(e) from e


