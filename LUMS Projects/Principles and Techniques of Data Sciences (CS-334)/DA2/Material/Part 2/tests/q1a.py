OK_FORMAT = True
test = {
    "name": "q1a",
    "points": 2,
    "suites": [
        {
            'cases': [
                {"code": r"""
                >>> assert nba_data_last_10.shape == (10, 22)
                """}
            ],
            'scored': True,
            'setup': "",
            'teardown': "",
            'type': 'doctest'
        }
    ]
}
