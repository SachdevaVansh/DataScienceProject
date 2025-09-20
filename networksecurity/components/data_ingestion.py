from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

## Configuration for Data Ingestion Config 

from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact

import os 
import sys 
import numpy as np 
import pandas as pd
import pymongo 
from typing import List 
from sklearn.model_selection import train_test_split

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

class DataIngestion:
    def __init__(self,data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e) from e

    def export_collection_as_dataframe(self):
        try:
            database_name=self.data_ingestion_config.database_name
            collection_name=self.data_ingestion_config.collection_name
            self.mongo_client=pymongo.MongoClient(MONGO_DB_URL)
            database=self.mongo_client[database_name]
            collection=database[collection_name]

            df=pd.DataFrame(list(collection.find()))
            if "_id" in df.columns:
                df=df.drop(columns=["_id"],axis=1)  
            df.replace({"na":np.nan},inplace=True)
            return df 

        except Exception as e:  
            raise NetworkSecurityException(e) from e
    
    def export_data_into_feature_store(self,dataframe:pd.DataFrame):
        try:
            feature_store_file_path=self.data_ingestion_config.feature_store_file_path
            dir_path=os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            dataframe.to_csv(feature_store_file_path,index=False,header=True)  
            return dataframe
        except Exception as e:
            raise NetworkSecurityException(e) from e

    def train_test_split(self,dataframe:pd.DataFrame):
        try:
            train_set,test_set=train_test_split(dataframe,test_size=self.data_ingestion_config.train_test_split_ratio,random_state=42)
            logging.info("Performed Train Test Split on the Data")
            logging.info(f"Train Set Length : {len(train_set)}")
            logging.info(f"Test Set Length : {len(test_set)}")

            train_dir=os.path.dirname(self.data_ingestion_config.training_file_path)
            test_dir=os.path.dirname(self.data_ingestion_config.testing_file_path)
            os.makedirs(train_dir,exist_ok=True)
            os.makedirs(test_dir,exist_ok=True)

            train_set.to_csv(self.data_ingestion_config.training_file_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path,index=False,header=True)

            logging.info("Exported Train and Test file path")

        except Exception as e:
            raise NetworkSecurityException(e) from e   

    def initiate_data_ingestion(self):
        try:
            dataframe=self.export_collection_as_dataframe()
            dataframe=self.export_data_into_feature_store(dataframe=dataframe)
            self.train_test_split(dataframe=dataframe)
            data_ingestion_artifact=DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
            return data_ingestion_artifact

        except Exception as e:  
            raise NetworkSecurityException(e) from e


