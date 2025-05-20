OK_FORMAT = True
test = {
    "name": "q4b",
    "points": 1,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> assert 'CarName' not in df.columns
                    >>> assert 'fuelsystem' not in df.columns
                    >>> assert 'cylindernumber' not in df.columns
                    >>> assert len(df.columns) > 0 
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
