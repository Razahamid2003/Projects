OK_FORMAT = True

test = {   'name': 'q10',
    'points': None,
    'suites': [   {   'cases': [   {'code': '>>> assert \'is_long\' in df.columns, "The column \'is_long\' does not exist in the DataFrame."\n', 'hidden': False, 'locked': False},
                                   {   'code': '>>> expected_counts = {0: 661, 1: 123}\n'
                                               ">>> counts = df['is_long'].value_counts().to_dict()\n"
                                               '>>> assert counts == expected_counts, f"The value counts of \'is_long\' are not as expected. Got {counts}, expected {expected_counts}."\n',
                                       'hidden': False,
                                       'locked': False},
                                   {'code': '>>> \n', 'hidden': False, 'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
