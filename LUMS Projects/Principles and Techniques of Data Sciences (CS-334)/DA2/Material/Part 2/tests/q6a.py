OK_FORMAT = True
test = {
    "name": "q6a",
    "points": 4,
    "suites": [
        {
            'cases': [
                {
                    "code": r"""
                    >>> # Check the shape of the avg_stats_pivot DataFrame
                    >>> expected_shape = (30, 3)
                    >>> assert avg_stats_pivot.shape == expected_shape, f"Expected shape {expected_shape}, but got {avg_stats_pivot.shape}."
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
