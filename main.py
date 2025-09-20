from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig,DataValidationConfig
import sys 

if __name__=="__main__":
    try:
        training_pipeline_config=TrainingPipelineConfig()
        data_ingestion_config=DataIngestionConfig(training_pipeline_config)
        data_ingestion=DataIngestion(data_ingestion_config)
        logging.info("Starting Data Ingestion")
        data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
        print(data_ingestion_artifact)
        logging.info("Data Ingestion Completed")

        data_validation_config=DataValidationConfig(training_pipeline_config)
        data_validation=DataValidation(data_validation_config,data_ingestion_artifact)
        logging.info("Starting the Data Validation")
        data_validation_artifact=data_validation.initiate_data_validation()
        print(data_validation_artifact)
        logging.info("Data Validation Completed")

    except Exception as e:
        raise NetworkSecurityException(e) from e