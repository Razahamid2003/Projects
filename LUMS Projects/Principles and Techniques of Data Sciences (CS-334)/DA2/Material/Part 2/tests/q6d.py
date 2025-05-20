OK_FORMAT = True
test = {
    "name": "q6d",
    "points": 4,
    "suites": [
        {
            'cases': [
                {
                    "code": r"""
                    >>> # Check that the country with the most players for the team with the highest average assists is correct
                    >>> expected_country_most_players = 'USA'
                    >>> assert country_most_players == expected_country_most_players, f"Expected country with most players {expected_country_most_players}, but got {country_most_players}."
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
