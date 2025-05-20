OK_FORMAT = True
test = {
    "name": "q3",
    "points": 2,
    "suites": [
        {
            'cases': [
                {
                    "code": r"""
                    >>> # Check the shape of the performance_summary DataFrame
                    >>> assert performance_summary.shape == (8, 4), f"Expected shape (8, 4), but got {performance_summary.shape}."
                    
                    >>> # Check that the expected columns are present
                    >>> expected_columns = ['points', 'rebounds', 'assists', 'age']
                    >>> assert all(col in performance_summary.columns for col in expected_columns), "Performance summary should include all expected columns."
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
