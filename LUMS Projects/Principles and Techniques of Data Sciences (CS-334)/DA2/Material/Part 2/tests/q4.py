OK_FORMAT = True
test = {
    "name": "q4",
    "points": 4,
    "suites": [
        {
            'cases': [
                {
                    "code": r"""
                    >>> # Check that the top_3_teams list contains the expected team abbreviations
                    >>> expected_teams = ['NOP', 'GSW', 'LAL']
                    >>> assert top_3_teams == expected_teams, f"Expected top teams {expected_teams}, but got {top_3_teams}."
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
