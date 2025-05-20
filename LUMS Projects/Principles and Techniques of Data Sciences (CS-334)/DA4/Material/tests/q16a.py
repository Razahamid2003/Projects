OK_FORMAT = True
test = {
    "name": "q16a",
    "points": 4,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> # Example 1: Perfect predictions (R² = 1)
                    >>> actual = [3, -0.5, 2, 7]
                    >>> predicted = [3, -0.5, 2, 7]
                    >>> result = calculateR2score(actual, predicted)
                    >>> assert abs(result - 1) < 1e-6, f"Expected R² = 1, but got {result}"
                    
                    >>> # Example 2: Predictions equal to the mean of actual values (R² = 0)
                    >>> actual = [1, 2, 3, 4, 5]
                    >>> predicted = [3, 3, 3, 3, 3]
                    >>> result = calculateR2score(actual, predicted)
                    >>> assert abs(result - 0) < 1e-6, f"Expected R² = 0, but got {result}"
                    
                    >>> # Example 3: Partial fit (known R² value)
                    >>> actual = [1, 2, 3]
                    >>> predicted = [2, 2, 2]
                    >>> result = calculateR2score(actual, predicted)
                    >>> expected_r2 = 1 - (2 / 2)  # Expected R² = 0
                    >>> assert abs(result - expected_r2) < 1e-6, f"Expected R² = {expected_r2}, but got {result}"
                    
                    >>> # Example 4: Large dataset (random values)
                    >>> import numpy as np
                    >>> np.random.seed(42)
                    >>> actual = np.random.rand(100)
                    >>> predicted = actual + np.random.normal(0, 0.1, 100)  # Add noise
                    >>> result = calculateR2score(actual, predicted)
                    >>> assert 0 <= result <= 1, f"Expected R² to be between 0 and 1, but got {result}"
                    """
                }
            ],
            "scored": True,
            "setup": "",
            "teardown": "",
            "type": "doctest"
        }
    ]
}
