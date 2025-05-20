OK_FORMAT = True
test = {
    "name": "q17a",
    "points": 3,
    "suites": [
        {
            "cases": [
                {
                    "code": r"""
                    >>> import numpy as np
                    >>> import pandas as pd
                    >>> # Example 1: Simple dataset with known parameters
                    >>> X = pd.DataFrame([[1, 1], [1, 2], [1, 3]])
                    >>> y = pd.Series([1, 2, 3])
                    >>> result = optimal_params(X, y)
                    >>> expected = np.array([0, 1])  # Manually computed thetas for this dataset
                    >>> np.testing.assert_almost_equal(result, expected, decimal=6, err_msg="Optimal parameters are incorrect for Example 1")
                    
                    >>> # Example 2: Dataset with larger dimensions
                    >>> X = pd.DataFrame([[1, 0], [0, 1], [1, 1], [1, 2]])
                    >>> y = pd.Series([1, 2, 3, 4])
                    >>> result = optimal_params(X, y)
                    >>> expected = np.linalg.inv(X.T @ X) @ X.T @ y.to_numpy()
                    >>> np.testing.assert_almost_equal(result, expected, decimal=6, err_msg="Optimal parameters are incorrect for Example 2")
                    
                    >>> # Example 3: Ensure output dimensions match
                    >>> X = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
                    >>> y = pd.Series([1, 2, 3])
                    >>> result = optimal_params(X, y)
                    >>> assert result.shape == (X.shape[1],), f"Output shape {result.shape} does not match expected shape {(X.shape[1],)}"
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
