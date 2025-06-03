import pandas as pd

def load_dataset(dataset=None, sep=","):
        """
        Load dataset from a file or DataFrame.
        """
        if dataset:
            if isinstance(dataset, str):
                if dataset.endswith((".csv", ".txt", ".tsv")):
                    data = pd.read_csv(dataset, sep=sep)
                elif dataset.endswith(".xlsx"):
                    data = pd.read_excel(dataset)
                elif dataset.endswith(".json"):
                    with open(dataset, "r", encoding="utf-8") as f:
                        data = pd.read_json(f)
                else:
                    raise ValueError(f"Unrecognized format of {dataset}")
            elif isinstance(dataset, pd.DataFrame):
                data = dataset
            else:
                raise TypeError("Dataset must be a path or a pandas DataFrame.")
        else:
            raise ValueError("Dataset cannot be None.")
        return data