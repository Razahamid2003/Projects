OK_FORMAT = True
test = {
    "name": "q2b",
    "points": 2,
    "suites": [
        {
            'cases': [
                {
                    "code": r"""
                    >>> # Check that missing_df has the correct columns
                    >>> assert list(missing_df.columns) == ['Total No. of Missing Values', '% of Missing Values']
                    
                    >>> # Check that missing_df has the expected number of rows (equal to the number of columns in nba_data)
                    >>> assert missing_df.shape[0] == nba_data.shape[1]
                    
                    >>> # Check that the '% of Missing Values' column is correctly calculated
                    >>> assert (missing_df['% of Missing Values'] == round((missing_df['Total No. of Missing Values'] / len(nba_data)) * 100, 2)).all()
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
