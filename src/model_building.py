import os
import pickle
import mlflow
import dagshub
from flaml import AutoML
from sklearn.metrics import accuracy_score


def model_building(X_train, X_test, y_train, y_test, preprocessor):

    dagshub.init(
        repo_owner='KhushiRai2009',
        repo_name='Diabetes_predictionmodel',
        mlflow=True
    )

    mlflow.set_experiment("Diabetes_AutoML")

    with mlflow.start_run():

        automl = AutoML()

        settings = {
            "time_budget": 300,
            "metric": "accuracy",
            "task": "classification",
            "estimator_list": ["rf","lgbm","xgboost","extra_tree","lrl2","svc"]
        }

        automl.fit(X_train=X_train, y_train=y_train, **settings)

        y_pred = automl.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print("Best Model:", automl.model)
        print("Accuracy:", acc)

        mlflow.log_metric("accuracy", acc)

        # -------- SAVE MODEL --------
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        model_dir = os.path.join(BASE_DIR,"Deployment", "models")
        os.makedirs(model_dir, exist_ok=True)

        file_path = os.path.join(model_dir, "diabetes_model.pkl")

        print("Reached saving section")
        print("BASE_DIR:",BASE_DIR)
        print("Model dir:",model_dir)
        print("File path:",file_path)

        model_package = {
            "model": automl.model,
            "preprocessor": preprocessor,
            "accuracy": acc
        }

        print("Saving model to:", file_path)

        with open(file_path, "wb") as f:
            pickle.dump(model_package, f)

        print("Model saved successfully!")


    return automl.model