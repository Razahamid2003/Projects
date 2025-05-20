OK_FORMAT = True
test = {
    "name": "q1b",
    "points": 2,
    "suites": [
        {
            'cases': [
                {
                    "code": r"""
                    >>> assert list(nba_data.columns) == ['Unnamed: 0', 'playerName', 'teamAbbreviation', 'age', 
                    ...                                        'playerHeight', 'playerWeight', 'college', 'country', 
                    ...                                        'draftYear', 'draftRound', 'draftNumber', 'gamesPlayed', 
                    ...                                        'points', 'rebounds', 'assists', 'netRating', 
                    ...                                        'offensiveReboundPercentage', 'defensiveReboundPercentage', 
                    ...                                        'usagePercentage', 'trueShootingPercentage', 'assistPercentage', 
                    ...                                        'season']
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
