OK_FORMAT = True
test = {
    "name": "q5",
    "points": 6,
    "suites": [
        {
            'cases': [
                {
                    "code": r"""
                    >>> # Check that the count of player names starting with 'Mc' or 'O'' is as expected
                    >>> expected_count = 300
                    >>> assert mc_o_names_count == expected_count, f"Expected count {expected_count}, but got {mc_o_names_count}."
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
