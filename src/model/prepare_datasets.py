import click
import pandas as pd
from sklearn.model_selection import train_test_split


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("train_output_path", type=click.Path())
@click.argument("test_output_path", type=click.Path())
@click.argument("test_size", type=float)
def prepare_dataset(input_path: str, train_output_path: str, test_output_path: str, test_size: float):
    """
    Split dataset into train and test sets.

    INPUT_PATH: path to dataset with features  
    TRAIN_OUTPUT_PATH: path to save train set  
    TEST_OUTPUT_PATH: path to save test set  
    TEST_SIZE: proportion of dataset to use as test set (e.g., 0.2)
    """

    df = pd.read_csv(input_path)

    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=df['target'] if 'target' in df.columns else None
    )

    train.to_csv(train_output_path, index=False)
    test.to_csv(test_output_path, index=False)

    click.echo(f"Train set saved to {train_output_path}, test set saved to {test_output_path}")


if __name__ == "__main__":
    prepare_dataset()
