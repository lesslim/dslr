from .dataframe import DataFrame


def read_csv(*args, **kwargs) -> DataFrame:
    """
    Solely for compatibility with pandas. For better auto-completion and
    documentation use bears.DataFrame.read_csv.
    """
    return DataFrame.read_csv(*args, **kwargs)  # type: ignore
