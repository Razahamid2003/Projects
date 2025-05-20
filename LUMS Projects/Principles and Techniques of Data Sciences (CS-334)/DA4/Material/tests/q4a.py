OK_FORMAT = True
test = {
    "name": "q4a",
    "points": 1,
    "suites": [
        {
            'cases': [
                {
                    "code": r"""
                    >>> import pandas as pd
                    >>> expected_missing = df.isnull().sum()
                    >>> pd.testing.assert_series_equal(missing, expected_missing)
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
