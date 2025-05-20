OK_FORMAT = True

test = {
    "name": "test_q5",
    "points": 1,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> # Expected value for DiD_homicide
                    >>> expected_did_homicide = 0.13131605353037656
                    >>> # Define tolerance for floating-point comparison
                    >>> tolerance = 1e-10
                    >>> # Check if DiD_homicide is defined and matches the expected value within the tolerance
                    >>> assert 'DiD_homicide' in globals(), "DiD_homicide variable is not defined."
                    >>> assert abs(DiD_homicide - expected_did_homicide) <= tolerance, f"DiD_homicide value {DiD_homicide} is not within the expected range."
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
