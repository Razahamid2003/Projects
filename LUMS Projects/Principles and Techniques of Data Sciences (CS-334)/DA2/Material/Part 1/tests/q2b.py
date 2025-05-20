OK_FORMAT = True

test = {   'name': 'q2b',
    'points': None,
    'suites': [   {   'cases': [   {   'code': ">>> meta_score_desc = df['Meta_score'].describe()\n>>> assert round(meta_score_desc['mean'], 5) == 77.97153, 'Mean should be 77.97153'\n",
                                       'hidden': False,
                                       'locked': False},
                                   {   'code': ">>> meta_score_desc = df['Meta_score'].describe()\n>>> assert round(meta_score_desc['std'], 5) == 11.36206, 'Standard deviation should be 11.36206'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
