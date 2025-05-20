OK_FORMAT = True

test = {   'name': 'q9a',
    'points': None,
    'suites': [   {   'cases': [   {'code': ">>> assert set(star_df.columns) == {'star', 'gross'}, 'Columns do not match'\n", 'hidden': False, 'locked': False},
                                   {   'code': ">>> aamir_gross = star_df[star_df['star'] == 'Aamir Khan']['gross'].values[0]\n"
                                               '>>> assert aamir_gross == 33332120.0, "Aamir Khan\'s gross value does not match"\n',
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
