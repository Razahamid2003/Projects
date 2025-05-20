OK_FORMAT = True

test = {   'name': 'q4c',
    'points': None,
    'suites': [   {   'cases': [   {   'code': '>>> expected_value_counts = {2: 200, 0: 180, 1: 163, 3: 135, 6: 39, 7: 24, 5: 23, 4: 10, 10: 6, 9: 2, 8: 1, 11: 1, 12: 1}\n'
                                               ">>> actual_value_counts = df['Certificate'].value_counts().sort_index()\n"
                                               '>>> for key, expected_count in expected_value_counts.items():\n'
                                               "...     assert actual_value_counts.get(key, 0) == expected_count, f'Expected count for {key} is {expected_count}, but got "
                                               "{actual_value_counts.get(key, 0)}.'\n",
                                       'hidden': False,
                                       'locked': False},
                                   {'code': '>>> assert df[\'Certificate\'].dtype == \'int64\', "The data type of \'Certificate\' should be int64."\n', 'hidden': False, 'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
