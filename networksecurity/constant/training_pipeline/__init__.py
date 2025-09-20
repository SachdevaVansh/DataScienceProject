import os 
import sys
import numpy as np 
import pandas as pd 

TARGET_COLUMN = "Result"
PIPELINE_NAME:str= "NetworkSecurity"
ARTIFACT_DIR: str= "Artifacts"
FILE_NAME: str= "phishingData.csv"

TRAIN_FILE_NAME: str= "train.csv"
TEST_FILE_NAME: str= "test.csv"

#Data Ingestion related constants
DATA_INGESTION_COLLECTION_NAME: str= "PhishingData"
DATA_INGESTION_DATABASE_NAME : str= "NetworkSecurity"
DATA_INGESTION_DIR_NAME: str= "data_ingestion"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float= 0.2      
DATA_INGESTION_FEATURE_STORE_DIR: str= "feature_store"
DATA_INGESTION_INGESTED_DIR: str="ingested"

# Data Schema related constants(Validation)
SCHEMA_FILE_PATH: str= os.path.join("data_schema","schema.yaml")

# Data Validation related constants
DATA_VALIDATION_DIR_NAME: str="data_validation"
DATA_VALIDATION_VALID_DIR: str="validated"
DATA_VALIDATION_INVALID_DIR: str="invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str="drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str="report.yaml"

# Data Transformation related constants
DATA_TRANSFORMATION_DIR_NAME: str="data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str="transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str="transformed_object"

# KNN Imputer related constants to replace Nan Values
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict={
    "missing_values":np.nan,
    "n_neighbors":3,
    "weights":"uniform",
}
# For the Preprocessing.pkl file
PREPROCESSED_OBJECT_FILE_NAME: str="preprocessed.pkl"

# NUmpy array file paths 
DATA_TRANSFORMATION_TRAIN_FILE_PATH: str = "train.npy"
DATA_TRANSFORMATION_TEST_FILE_PATH: str = "test.npy"