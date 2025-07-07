import pandas as pd


def rename_datasets_text_features(X_train: pd.DataFrame,
                                  X_test: pd.DataFrame,
                                  Y_train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Rename the columns of the textual and labels datasets. The changes are :
        `designation` -> `title`
        `prdtypecode` -> `labels`

    Args:
        X_train (pd.DataFrame): The training dataset.
        X_test (pd.DataFrame): The testing dataset.
        Y_train (pd.DataFrame): The training labels dataset.

    Returns:
        tuple: The three datasets with their new column names.
    """
    X_train.rename(columns={"designation": "title"}, inplace=True)
    X_test.rename(columns={"designation": "title"}, inplace=True)
    Y_train.rename(columns={"prdtypecode": "category"}, inplace=True)
    return (X_train, X_test, Y_train)



def remove_special_characters(df: pd.DataFrame) -> pd.DataFrame: ...
