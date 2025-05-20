OK_FORMAT = True

test = {   'name': 'q6b',
    'points': None,
    'suites': [   {   'cases': [   {   'code': '>>> expected_unique_values = [1990.0, 1970.0, 2000.0, 1950.0, 2010.0, 1960.0, 1980.0, 1940.0, 1930.0, 1920.0]\n'
                                               ">>> unique_values = df['released_year'].unique()\n"
                                               ">>> assert all((value in unique_values for value in expected_unique_values)), f'Expected unique values {expected_unique_values}, but got "
                                               "{unique_values}.'\n",
                                       'hidden': False,
                                       'locked': False},
                                   {   'code': ">>> expected_shape = (784, 35)\n>>> assert df.shape == expected_shape, f'Expected shape {expected_shape}, but got {df.shape}.'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
