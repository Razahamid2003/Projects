OK_FORMAT = True
test = {
    "name": "",
    "points": 1,
    "suites": [
        {
            "cases": [ 
                {
                    "code": r"""
                    >>> assert all(isinstance(age, (int, float)) for age in df['student_age'])
                    >>> assert df[df['student_age'] == 20.5].shape[0] > 0
                    >>> assert df[df['student_age'] == 18].shape[0] > 0
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
