import itertools

OK_FORMAT = True

# Expected outliers for each team
expected_BOS_outliers = ['Jaylen Brown', 'Jayson Tatum']
expected_GSW_outliers = ['Stephen Curry', 'Klay Thompson', 'Jordan Poole']
expected_MIA_outliers = ['Tyler Herro', 'Bam Adebayo', 'Jimmy Butler']

# Manual method to generate permutations
def generate_permutations(lst):
    if len(lst) == 0:
        return [[]]
    permutations = []
    for i in range(len(lst)):
        current = lst[i]
        remaining = lst[:i] + lst[i+1:]
        for p in generate_permutations(remaining):
            permutations.append([current] + p)
    return permutations

# Generate permutations of expected outliers
BOS_permutations = generate_permutations(expected_BOS_outliers)
GSW_permutations = generate_permutations(expected_GSW_outliers)
MIA_permutations = generate_permutations(expected_MIA_outliers)

test = {
    "name": "q8c",
    "points": 8,
    "suites": [
        {
            'cases': [
                {
                    "code": f"""
                    >>> # Check for Boston Celtics outliers against all permutations
                    >>> assert any(BOS_outliers == list(perm) for perm in {BOS_permutations}), "BOS outliers do not match any expected values"

                    >>> # Check for Golden State Warriors outliers against all permutations
                    >>> assert any(GSW_outliers == list(perm) for perm in {GSW_permutations}), "GSW outliers do not match any expected values"

                    >>> # Check for Miami Heat outliers against all permutations
                    >>> assert any(MIA_outliers == list(perm) for perm in {MIA_permutations}), "MIA outliers do not match any expected values"
                    """,
                    'scored': True,
                    'setup': "",
                    'teardown': "",
                    'type': 'doctest'
                }
            ],
            'scored': True,
            'setup': "",
            'teardown': "",
            'type': 'doctest'
        }
    ]
}
