OK_FORMAT = True
test = {
    "name": "q2c",
    "points": 2,
    "suites": [
        {
            'cases': [
                {
                    "code": r"""
                    >>> # Define the expected lists for comparison
                    >>> expected_numerical_data = ['age', 'playerHeight', 'playerWeight', 'gamesPlayed', 'points', 
                    ...                             'rebounds', 'assists', 'netRating', 'offensiveReboundPercentage', 
                    ...                             'defensiveReboundPercentage', 'usagePercentage', 
                    ...                             'trueShootingPercentage', 'assistPercentage']
                    
                    >>> expected_categorical_data = ['playerName', 'teamAbbreviation', 'college', 'country', 
                    ...                              'draftYear', 'draftRound', 'draftNumber', 'season']
                    
                    >>> # Check that numerical_data contains the correct numerical columns
                    >>> assert numerical_data == expected_numerical_data, f"Expected numerical columns: {expected_numerical_data}, but got: {numerical_data}"
                    
                    >>> # Check that categorical_data contains the correct categorical columns
                    >>> assert categorical_data == expected_categorical_data, f"Expected categorical columns: {expected_categorical_data}, but got: {categorical_data}"
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
