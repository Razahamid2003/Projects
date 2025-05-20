OK_FORMAT = True

test = {
    "name": "test_q3a",
    "points": 1,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> # Check if DF_CATE_STATE is defined
                    >>> assert 'DF_CATE_STATE' in globals(), "DF_CATE_STATE is not defined."
                    >>> # Calculate mean and std for ATE_Homicide
                    >>> mean_ate = DF_CATE_STATE['ATE_Homicide'].mean()
                    >>> std_ate = DF_CATE_STATE['ATE_Homicide'].std()
                    >>> # Expected values for mean and std
                    >>> expected_mean = -0.20842067520786078
                    >>> expected_std = 0.7533303780013819
                    >>> # Tolerances
                    >>> tolerance = 1e-5
                    >>> # Check mean and std against expected values
                    >>> assert abs(mean_ate - expected_mean) <= tolerance, f"Mean ATE_Homicide {mean_ate} is not within the expected range."
                    >>> assert abs(std_ate - expected_std) <= tolerance, f"STD ATE_Homicide {std_ate} is not within the expected range."
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
