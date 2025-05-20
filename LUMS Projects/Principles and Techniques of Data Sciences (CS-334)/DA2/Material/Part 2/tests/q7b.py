OK_FORMAT = True
test = {
    "name": "q7b",
    "points": 4,
    "suites": [
        {
            'cases': [
                {
                    "code": r"""
                    >>> # Check that the correlation coefficient is as expected
                    >>> expected_correlation_coefficient = 0.83
                    >>> assert correlation_coefficient == expected_correlation_coefficient, f"Expected correlation coefficient {expected_correlation_coefficient}, but got {correlation_coefficient}."
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
