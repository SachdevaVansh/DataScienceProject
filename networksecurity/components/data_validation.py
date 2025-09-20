import os 
import sys 

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH

from networksecurity.utils.utils import read_yaml_file, write_yaml_file

import pandas as pd 
import numpy as np
from scipy.stats import ks_2samp

class DataValidation:
    def __init__(self,data_validation_config:DataValidationConfig,
                 data_ingestion_artifact:DataIngestionArtifact):
        try:
            self.data_validation_config=data_validation_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self.schema_config=read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e) from e

    @staticmethod 
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e) from e
    
    def validate_number_of_columns(self,dataframe:pd.DataFrame)->bool:
        try:
            number_of_columns=len(self.schema_config['columns'])
            logging.info(f"Number of Columns in Schema : {number_of_columns}")
            logging.info(f"Number of Columns in Dataframe : {len(dataframe.columns)}")
            if len(dataframe.columns)==number_of_columns:
                return True
            return False
        except Exception as e:  
            raise NetworkSecurityException(e) from e

    def detect_numerical_columns(self,dataframe:pd.DataFrame)->list:
        try:
            numerical_columns=self.schema_config['numerical_columns']
            dataframe_columns=dataframe.columns
            missing_numerical_columns=[]  
            for num_column in numerical_columns:
                if num_column not in dataframe_columns:
                    missing_numerical_columns.append(num_column)
            return missing_numerical_columns
        except Exception as e:
            raise NetworkSecurityException(e) from e  

    def detect_data_drift(self,base_df,current_df,threshold=0.05)->bool:
        try:
            status=True
            report={}
            for column in base_df.columns:
                d1=base_df[column]
                d2=current_df[column]
                ks_test_statistic,p_value=ks_2samp(d1,d2)
                if threshold<=p_value:
                    drift=False
                else:
                    drift=True
                    status=False
                report.update({column:{"p_value":float(p_value),
                                       "drift_status":drift}})
            drift_report_file_path=self.data_validation_config.drift_report_file_path

            dir_path=os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report, replace=True)
            return status 

        except Exception as e:
            raise NetworkSecurityException(e) from e


    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            logging.info("Starting the Data Validation")

            #Read the train and test file path 
            train_file_path=self.data_ingestion_artifact.train_file_path 
            test_file_path=self.data_ingestion_artifact.test_file_path
            logging.info(f"Train File Path : {train_file_path}")
            logging.info(f"Test File Path : {test_file_path}")

            # Read the data from train and test file path 
            train_dataframe=DataValidation.read_data(train_file_path)
            test_dataframe=DataValidation.read_data(test_file_path)

            ##validate the number of columns
            status_train=self.validate_number_of_columns(dataframe=train_dataframe)
            if not status_train:
                error_message=f"Train DataFrame does not contain all columns"
                logging.info(error_message)
                
            status_test=self.validate_number_of_columns(dataframe=test_dataframe)
            if not status_test:
                error_message=f"Test DataFrame does not contain all columns"
                logging.info(error_message)

            if status_train and status_test:
                validation_status=True
                logging.info("Both Train and Test Dataframe contain all columns")

            #validate the numerical columns
            missing_numerical_columns_train=self.detect_numerical_columns(train_dataframe)
            if len(missing_numerical_columns_train)>0:
                logging.info(f"Missing Numerical Columns in Train Dataframe : {missing_numerical_columns_train}")
                raise Exception(f"Missing Numerical Columns in Train Dataframe : {missing_numerical_columns_train}")
            
            missing_numerical_columns_test=self.detect_numerical_columns(test_dataframe)
            if len(missing_numerical_columns_test)>0:
                logging.info(f"Missing Numerical Columns in Test Dataframe : {missing_numerical_columns_test}")
                raise Exception(f"Missing Numerical Columns in Test Dataframe : {missing_numerical_columns_test}")

            if (len(missing_numerical_columns_train)==0) and (len(missing_numerical_columns_test)==0):    
                logging.info("No Missing Numerical Columns in both Train and Test Dataframe")

            #Data Drift Detection
            drift_status_train=self.detect_data_drift(base_df=train_dataframe,current_df=test_dataframe)
            if drift_status_train:
                logging.info("No Data Drift detected between Train and Test Dataframe")

                dir_path=os.path.dirname(self.data_validation_config.valid_train_file_path)
                os.makedirs(dir_path, exist_ok=True)

                train_dataframe.to_csv(self.data_validation_config.valid_train_file_path,index=False,header=True)

                dir_path=os.path.dirname(self.data_validation_config.valid_test_file_path)
                os.makedirs(dir_path, exist_ok=True)
                test_dataframe.to_csv(self.data_validation_config.valid_test_file_path,index=False,header=True)
            else:
                logging.info("Data Drift detected between Train and Test Dataframe")
            
            #Create Data Validation Artifact
            data_validation_artifact=DataValidationArtifact(
                validation_status=drift_status_train,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,  
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )
            return data_validation_artifact

        except Exception as e:
            raise NetworkSecurityException(e) from e


