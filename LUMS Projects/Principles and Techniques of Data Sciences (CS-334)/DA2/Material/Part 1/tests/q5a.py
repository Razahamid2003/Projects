OK_FORMAT = True

test = {   'name': 'q5a',
    'points': None,
    'suites': [   {   'cases': [   {   'code': ">>> expected_genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'History', "
                                               "'Horror', 'Music', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'War', 'Western']\n"
                                               '>>> for genre in expected_genres:\n'
                                               '...     assert genre in df.columns, f"Column for genre \'{genre}\' is missing."\n',
                                       'hidden': False,
                                       'locked': False},
                                   {   'code': ">>> expected_genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'History', "
                                               "'Horror', 'Music', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'War', 'Western']\n"
                                               '>>> for index, row in df.iterrows():\n'
                                               '...     for genre in unique_genres:\n'
                                               "...         expected_value = 1 if genre in row['Genre'].split(', ') else 0\n"
                                               '...         assert row[genre] == expected_value, f"Value for genre \'{genre}\' at index {index} is incorrect."\n',
                                       'hidden': True,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
