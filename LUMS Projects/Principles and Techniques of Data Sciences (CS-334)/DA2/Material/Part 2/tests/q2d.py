OK_FORMAT = True
test = {
    "name": "q2d",
    "points": 2,
    "suites": [
        {
            'cases': [
                {
                    "code": r"""
                    
                    >>> # Check that there are no null values in the 'college' column
                    >>> assert nba_data['college'].isnull().sum() == 0, "Test failed: There are still null values in 'college'."
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
