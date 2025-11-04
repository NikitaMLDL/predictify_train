import click
import pandas as pd
from sklearn.metrics import accuracy_score
import joblib
import mlflow
from src import init_mlflow


@click.command()
@click.argument("test_path", type=click.Path(exists=True))
@click.argument("model_path", type=click.Path(exists=True))
def evaluate(test_path: str, model_path: str):
    """
    Evaluate trained model on test set.

    TEST_PATH: path to test CSV
    MODEL_PATH: path to trained model
    """
    init_mlflow()
    mlflow.set_experiment("auto_ml_experiment")
    df = pd.read_csv(test_path)
    X = df.drop(columns=['target'])
    y = df['target']

    # Загружаем модель
    clf = joblib.load(model_path)

    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)

    with mlflow.start_run():
        mlflow.log_metric("accuracy", acc)

    click.echo(f"Accuracy on test set: {acc:.4f}!")


if __name__ == "__main__":
    evaluate()
