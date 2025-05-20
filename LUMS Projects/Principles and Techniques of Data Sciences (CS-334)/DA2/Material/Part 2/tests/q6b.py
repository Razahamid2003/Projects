OK_FORMAT = True
test = {
    "name": "q6b",
    "points": 6,
    "suites": [
        {
            'cases': [
                {
                    "code": r"""
                    >>> # Check that the teams with the highest average points, rebounds, and assists are correct
                    >>> expected_highest_avg_points_team = 'PHX'
                    >>> expected_highest_avg_rebounds_team = 'NYK'
                    >>> expected_highest_avg_assists_team = 'PHX'

                    >>> assert highest_avg_points_team == expected_highest_avg_points_team, f"Expected team with highest average points {expected_highest_avg_points_team}, but got {highest_avg_points_team}."
                    >>> assert highest_avg_rebounds_team == expected_highest_avg_rebounds_team, f"Expected team with highest average rebounds {expected_highest_avg_rebounds_team}, but got {highest_avg_rebounds_team}."
                    >>> assert highest_avg_assists_team == expected_highest_avg_assists_team, f"Expected team with highest average assists {expected_highest_avg_assists_team}, but got {highest_avg_assists_team}."
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
