OK_FORMAT = True
test = {
    "name": "q8",
    "points": 3,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> # Columns that should be one-hot encoded
                    >>> original_columns = ['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'enginetype']
                    >>> # Ensure original columns are removed
                    >>> for col in original_columns:
                    ...     assert col not in df.columns, f"Column {col} should have been removed after one-hot encoding"
                    >>> # Check for presence of one-hot encoded columns
                    >>> encoded_columns = [col for col in df.columns if any(oc in col for oc in original_columns)]
                    >>> assert len(encoded_columns) > 0, "No one-hot encoded columns found"
                    >>> # Verify that one-hot encoded columns contain only 0 and 1
                    >>> for col in encoded_columns:
                    ...     assert set(df[col].unique()).issubset({0, 1}), f"Column {col} contains non-binary values"
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
