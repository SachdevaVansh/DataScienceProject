import yaml 
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import os, sys 
import dill
import pickle
import numpy as np

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