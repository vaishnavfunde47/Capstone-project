import dagshub
import mlflow
import mlflow.sklearn
import json
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
import yaml
from src.logger import logger

mlflow.set_tracking_uri('https://dagshub.com/vaishnavfunde47/Capstone-project.mlflow')
dagshub.init(repo_owner='vaishnavfunde47', repo_name='Capstone-project', mlflow=True)

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train the Logistic Regression model."""
    try:
        clf = LogisticRegression(C=1, solver='liblinear', penalty='l1')
        clf.fit(X_train, y_train)
        logger.info('Model training completed')
        return clf
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise


def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.info('Model saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:
        # Start an MLflow run
        with mlflow.start_run() as run:
            train_data = load_data('./data/processed/train_bow.csv')
            X_train = train_data.iloc[:, :-1].values
            y_train = train_data.iloc[:, -1].values
            
            # Parameters from your LogisticRegression
            C = 1.0
            solver = 'liblinear'
            penalty = 'l1'
            
            mlflow.log_param("C", C)
            mlflow.log_param("solver", solver)
            mlflow.log_param("penalty", penalty)

            clf = train_model(X_train, y_train)
            
            # Log the model to MLflow with artifact_path="model"
            mlflow.sklearn.log_model(clf, "model")
            
            # Save model locally if you still want the pickle file
            save_model(clf, 'models/model.pkl')
            
            # Save model_info.json so register_model.py can find the run_id and path
            model_info = {
                "run_id": run.info.run_id,
                "model_path": "model"
            }
            with open("model_info.json", "w") as f:
                json.dump(model_info, f)
                
            logger.info("Model building and MLflow logging completed successfully.")

    except Exception as e:
        logger.error(f'Failed to complete the model building process: {e}')
        print(f"Error: {e}")




if __name__ == '__main__':
    main()