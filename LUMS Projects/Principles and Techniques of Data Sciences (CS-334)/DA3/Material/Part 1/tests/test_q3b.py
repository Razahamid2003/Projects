OK_FORMAT = True

test = {
    "name": "test_q3b",
    "points": 1,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> # Expected CATE value
                    >>> expected_cate = -0.20842067520786078
                    >>> # Define tolerance for floating-point comparison
                    >>> tolerance = 1e-10
                    >>> # Check if CATE is defined and matches the expected value within the tolerance
                    >>> assert 'CATE' in globals(), "CATE variable is not defined."
                    >>> assert abs(CATE - expected_cate) <= tolerance, f"CATE value {CATE} is not within the expected range."
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
