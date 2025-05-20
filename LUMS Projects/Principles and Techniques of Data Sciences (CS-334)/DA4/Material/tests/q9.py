OK_FORMAT = True
test = {
    "name": "q9",
    "points": 3,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> # List of columns to normalize
                    >>> columns_to_normalize = ['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 'enginesize', 'horsepower', 'peakrpm']
                    >>> # Check that all columns to normalize are still in the DataFrame
                    >>> for col in columns_to_normalize:
                    ...     assert col in df.columns, f"Column {col} is missing after normalization"
                    >>> # Verify that all values in the normalized columns are between 0 and 1
                    >>> for col in columns_to_normalize:
                    ...     assert df[col].max() <= 1, f"Values in column {col} exceed 1"
                    ...     assert df[col].min() >= 0, f"Values in column {col} are less than 0"
                    >>> # Optionally, ensure other columns are unaffected
                    >>> other_columns = [col for col in df.columns if col not in columns_to_normalize]
                    >>> assert len(other_columns) > 0, "All columns were unexpectedly normalized"
                    """
                }
            ],
            "scored": True,
            "setup": "",
            "teardown": "",
            "type": "doctest"
        }
    ]
}
