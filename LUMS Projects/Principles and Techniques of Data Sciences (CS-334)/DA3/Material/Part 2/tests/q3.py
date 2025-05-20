OK_FORMAT = True
test = {
    "name": "",
    "points": 2,
    "suites": [
        {
            "cases": [ 
                {
                    "code": r"""
                    >>> assert all(data_type in [np.int64, np.int32, np.float64, int, float] for data_type in df[['gender', 'high_school_type', 'scholarship', 'sleep_quality', 'involvement_in_extracurriculars', 'attendance', 'assignments_completed', 'attended_tutorials', 'grade']].dtypes)
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
