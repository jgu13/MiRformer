import pandas as pd
import ast

def parse_mixed(x):
    # handle blanks/NA safely
    if x is None:
        return pd.NA
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return pd.NA

    # safely parse Python literals: -1 -> int(-1), "[(439, 445), (446, 452)]" -> list[tuple]
    val = ast.literal_eval(s)

    # normalize inner pairs to tuples (in case some rows are lists)
    if isinstance(val, list):
        return [tuple(p) for p in val]
    return val  # either -1 or something else already parsed

def load_dataset(dataset=None, sep=",", parse_seeds=False):
        """
        Load dataset from a file or DataFrame.
        """
        if dataset:
            if isinstance(dataset, str):
                if dataset.endswith((".csv", ".txt", ".tsv")):
                    if parse_seeds:
                        data = pd.read_csv(dataset, sep=sep, converters={"seeds": parse_mixed})
                    else:
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