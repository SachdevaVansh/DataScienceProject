import os, sys 
import numpy as np
import pandas as pd
import mlflow

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact

from networksecurity.utils.utils import load_numpy_array_data, save_object, load_object, evaluate_models
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier)

class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,
                 data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e) from e  

    def track_mlflow(self,best_model,classificationmetric):
        with mlflow.start_run():
            f1_score=classificationmetric.f1_score
            precision_score=classificationmetric.precision_score
            recall_score=classificationmetric.recall_score

            mlflow.log_metric("f1_score",f1_score)
            mlflow.log_metric("precision_score",precision_score)
            mlflow.log_metric("recall_score",recall_score)
            mlflow.sklearn.log_model(best_model,"model")

    def train_model(self,X_train,y_train,X_test,y_test):
        try:
            models={
                "Random Forest":RandomForestClassifier(verbose=1),
                "Decision Tree":DecisionTreeClassifier(),
                "Logistic Regression":LogisticRegression(verbose=1),
                "Gradient Boosting":GradientBoostingClassifier(verbose=1),
                "Ada Boost":AdaBoostClassifier()
            }

            params={
                "Decision Tree":{
                    'criterion':['gini','entropy','log_loss'],
                    #'splitter':['best','random'],
                    #'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    'criterion':['gini','entropy','log_loss'],
                    #'max_features':['sqrt','log2','log1p'],
                    #'n_estimators':[8,16,32,64,128,256],
                },
                "Logistic Regression":{
                    'penalty':['l1','l2','elasticnet','none'],
                    #'solver':['liblinear','newton-cg','lbfgs','saga'],
                },
                "Gradient Boosting":{
                    'loss':['log_loss','deviance','exponential'],
                    #'criterion':['friedman_mse','squared_error'],
                    #'max_features':['sqrt','log2'],
                    #'n_estimators':[8,16,32,64,128,256],
                },
                "Ada Boost":{
                    'algorithm':['SAMME','SAMME.R'],
                    #'n_estimators':[8,16,32,64,128,256],
                }
            }

            model_report:dict= evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                               models=models,params=params)

            ## Model report will be a dictionary with model name and its r2_score
            best_model_score= max(sorted(model_report.values()))

            ## Getting the best model
            best_model_name= list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            #Object of the best model
            best_model=models[best_model_name]

            y_train_pred= best_model.predict(X_train)
            ## Classification Metric 
            classification_train_metric=get_classification_score(y_true=y_train,y_pred=y_train_pred)

            ## Track the model (Train)
            self.track_mlflow(best_model,classification_train_metric)

            y_test_pred= best_model.predict(X_test)
            ## Classification Metric
            classification_test_metric=get_classification_score(y_true=y_test,y_pred=y_test_pred)

            ## Track the model (Test)
            self.track_mlflow(best_model,classification_test_metric)

            ## Loading the preprocessor pickle file 
            preprocessor=load_object(file_path=self.data_transformation_artifact.preprocessed_object_file_path)

            model_dir_path=os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            Network_model=NetworkModel(preprocessor=preprocessor, model=best_model)

            #Saving the best model i.e Network Model class 
            save_object(self.model_trainer_config.trained_model_file_path, obj=NetworkModel)

            ##Model Trainer Artifact 
            model_trainer_artifact=ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric)

            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e) from e
       
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            logging.info("Loading the transformed training and test array")
            train_arr=load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr=load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            logging.info("Splitting the training and test array into input and target feature")
            X_train,y_train=train_arr[:,:-1],train_arr[:,-1]
            X_test,y_test=test_arr[:,:-1],test_arr[:,-1]

            model_trainer_artifact=self.train_model(X_train,y_train,X_test,y_test)
            return model_trainer_artifact

        except Exception as e:

            raise NetworkSecurityException(e) from e
        