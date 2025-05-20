OK_FORMAT = True

test = {
    "name": "test_q2_2",
    "points": 1,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> assert 'p_value' in globals(), "p_value variable is not defined."
                    >>> assert abs(p_value - 9.195806210348483e-08) <= 1e-10, f"p_value {p_value} is not within the expected range."
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
