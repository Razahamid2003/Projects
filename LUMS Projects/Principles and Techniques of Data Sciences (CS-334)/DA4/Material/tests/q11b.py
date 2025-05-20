OK_FORMAT = True
test = {
    "name": "q11b",
    "points": 3,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> # Columns that should be dropped
                    >>> columns_to_drop = ['price', 'symboling', 'car_ID']
                    >>> # Verify train_x and test_x do not contain dropped columns
                    >>> for col in columns_to_drop:
                    ...     assert col not in train_x.columns, f"Column {col} is still in train_x"
                    ...     assert col not in test_x.columns, f"Column {col} is still in test_x"
                    >>> # Verify train_y and test_y only contain the 'price' column
                    >>> assert train_y.name == 'price', "train_y does not contain the 'price' column"
                    >>> assert test_y.name == 'price', "test_y does not contain the 'price' column"
                    >>> # Ensure row counts in train_x and train_y match
                    >>> assert len(train_x) == len(train_y), "Mismatch in rows between train_x and train_y"
                    >>> # Ensure row counts in test_x and test_y match
                    >>> assert len(test_x) == len(test_y), "Mismatch in rows between test_x and test_y"
                    >>> # Ensure train_x and test_x have no missing columns
                    >>> original_columns = set(train_df.columns) - set(columns_to_drop)
                    >>> assert set(train_x.columns) == original_columns, "train_x has unexpected columns dropped or added"
                    >>> assert set(test_x.columns) == original_columns, "test_x has unexpected columns dropped or added"
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
