import click
import pandas as pd


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
def clean_data(input_path: str, output_path: str):
    """
    Load raw dataset from INPUT_PATH, clean it, and save to OUTPUT_PATH.
    """

    df = pd.read_csv(input_path)

    df = df.drop_duplicates()

    df.to_csv(output_path, index=False)
    click.echo(f"Cleaned data saved to {output_path}")


if __name__ == "__main__":
    clean_data()
