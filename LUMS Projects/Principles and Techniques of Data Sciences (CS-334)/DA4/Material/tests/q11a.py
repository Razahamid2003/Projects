OK_FORMAT = True
test = {
    "name": "q11a",
    "points": 2,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> # Verify that train_df and test_df exist
                    >>> assert 'train_df' in globals(), "train_df is not defined"
                    >>> assert 'test_df' in globals(), "test_df is not defined"
                    >>> # Check that the total number of rows matches the original DataFrame
                    >>> assert len(train_df) + len(test_df) == len(df), "Row count mismatch between train_df and test_df"
                    >>> # Ensure there is no overlap between train_df and test_df
                    >>> train_indices = set(train_df.index)
                    >>> test_indices = set(test_df.index)
                    >>> assert train_indices.isdisjoint(test_indices), "train_df and test_df have overlapping rows"
                    >>> # Verify the split ratio
                    >>> expected_test_size = 0.25 * len(df)
                    >>> actual_test_size = len(test_df)
                    >>> assert abs(actual_test_size - expected_test_size) <= 1, f"Expected test_df size to be {expected_test_size}, but got {actual_test_size}"
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
