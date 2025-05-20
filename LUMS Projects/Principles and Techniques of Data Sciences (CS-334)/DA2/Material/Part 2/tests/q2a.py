OK_FORMAT = True
test = {
    "name": "q2a",
    "points": 2,
    "suites": [
        {
            'cases': [
                {
                    "code": r"""
                    >>> assert 'Unnamed: 0' not in nba_data.columns
                    """
                }
            ],
            'scored': True,
            'setup': "",
            'teardown': "",
            'type': 'doctest'
        }
    ]
}
