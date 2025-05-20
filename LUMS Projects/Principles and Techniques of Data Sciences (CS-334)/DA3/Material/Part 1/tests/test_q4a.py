OK_FORMAT = True

test = {
    "name": "test_q4a",
    "points": 1,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> # Check if DF_CATE_STATE_YEAR is defined
                    >>> assert 'DF_CATE_STATE_YEAR' in globals(), "DF_CATE_STATE_YEAR is not defined."
                    >>> # Calculate mean and std for ATE_Homicide
                    >>> mean_ate_year = DF_CATE_STATE_YEAR['ATE_Homicide'].mean()
                    >>> std_ate_year = DF_CATE_STATE_YEAR['ATE_Homicide'].std()
                    >>> # Expected values for mean and std
                    >>> expected_mean = -0.6419749809523808
                    >>> expected_std = 0.877713316382941
                    >>> # Tolerances
                    >>> tolerance = 1e-5
                    >>> # Check mean and std against expected values
                    >>> assert abs(mean_ate_year - expected_mean) <= tolerance, f"Mean ATE_Homicide {mean_ate_year} is not within the expected range."
                    >>> assert abs(std_ate_year - expected_std) <= tolerance, f"STD ATE_Homicide {std_ate_year} is not within the expected range."
                    """,
                    "hidden": False,
                    "locked": False,
                },
            ],
            "scored": True,
            "setup": "",
            "teardown": "",
            "type": "doctest"
        }
    ]
}
