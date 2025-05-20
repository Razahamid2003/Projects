OK_FORMAT = True

test = {   'name': 'q12',
    'points': None,
    'suites': [   {   'cases': [   {   'code': '>>> expected_possible_runtime = (102.0, 129.0)\n'
                                               ">>> assert isinstance(possible_runtime, tuple), 'possible_runtime should be a tuple.'\n"
                                               ">>> assert len(possible_runtime) == 2, 'possible_runtime should contain two elements: the 25th and 75th percentiles.'\n"
                                               ">>> assert np.isclose(possible_runtime[0], expected_possible_runtime[0], atol=0.01), f'Expected 25th percentile to be {expected_possible_runtime[0]}, "
                                               "but got {possible_runtime[0]}.'\n"
                                               ">>> assert np.isclose(possible_runtime[1], expected_possible_runtime[1], atol=0.01), f'Expected 75th percentile to be {expected_possible_runtime[1]}, "
                                               "but got {possible_runtime[1]}.'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
