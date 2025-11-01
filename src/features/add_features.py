import click
import pandas as pd
from sklearn.preprocessing import LabelEncoder


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
def add_features(input_path: str, output_path: str):
    """
    Add new features to the dataset.

    INPUT_PATH: path to cleaned data CSV
    OUTPUT_PATH: path to save dataset with new features
    """

    df = pd.read_csv(input_path)

    le = LabelEncoder()
    df['target'] = le.fit_transform(df['Species'])
    df.drop(columns=['Id', 'Species'], inplace=True)

    df['sepal_ratio'] = df['SepalLengthCm'] / df['SepalWidthCm']
    df['petal_ratio'] = df['PetalLengthCm'] / df['PetalWidthCm']

    df.to_csv(output_path, index=False)
    click.echo(f"Data with new features saved to {output_path}")


if __name__ == "__main__":
    add_features()
