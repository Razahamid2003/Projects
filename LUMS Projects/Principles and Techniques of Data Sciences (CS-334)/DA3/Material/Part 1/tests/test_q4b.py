OK_FORMAT = True

test = {
    "name": "test_q4b",
    "points": 1,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> # Expected CATE_Y value
                    >>> expected_cate_y = -0.6419749809523808
                    >>> # Define tolerance for floating-point comparison
                    >>> tolerance = 1e-10
                    >>> # Check if CATE_Y is defined and matches the expected value within the tolerance
                    >>> assert 'CATE_Y' in globals(), "CATE_Y variable is not defined."
                    >>> assert abs(CATE_Y - expected_cate_y) <= tolerance, f"CATE_Y value {CATE_Y} is not within the expected range."
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
