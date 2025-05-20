OK_FORMAT = True
test = {
    "name": "q6c",
    "points": 6,
    "suites": [
        {
            'cases': [
                {
                    "code": r"""
                    >>> # Check the shape of the players_count_pivot DataFrame
                    >>> expected_shape = (30, 44)
                    >>> assert players_count_pivot.shape == expected_shape, f"Expected shape {expected_shape}, but got {players_count_pivot.shape}."
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
