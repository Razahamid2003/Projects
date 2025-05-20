OK_FORMAT = True

test = {
    "name": "test_q6",
    "points": 1,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> # Define expected values for each DiD result
                    >>> expected_values = {
                    ...     'DiD_robbery': -2.583372701149443,
                    ...     'DiD_larceny': -0.6787267651893671,
                    ...     'DiD_assault': -4.630575024630616,
                    ...     'DiD_murder': 0.15640436305418648
                    ... }
                    >>> # Define tolerance for floating-point comparison
                    >>> tolerance = 1e-10
                    >>> # Check each DiD variable
                    >>> for var_name, expected_value in expected_values.items():
                    ...     assert var_name in globals(), f"{var_name} variable is not defined."
                    ...     actual_value = globals()[var_name]
                    ...     assert abs(actual_value - expected_value) <= tolerance, f"{var_name} value {actual_value} is not within the expected range."
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
