OK_FORMAT = True

test = {
    "name": "test_q2_1",
    "points": 1,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> assert 'ATE' in globals(), "ATE variable is not defined."
                    >>> assert abs(ATE - 1.3488435295735863) <= 1e-3, f"ATE value {ATE} is not within the expected range."
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
