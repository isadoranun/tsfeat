from ipy_table import *

import pandas
import numpy as np

from Feature import FeatureSpace


def Table(a):

    # all_features = np.random.uniform(size=(len(a.result(method='array')),1),
        #low=1, high = 2)

    # all_features[:,0] = a.result(method= 'array')
    # df = pandas.DataFrame(all_features)
    # df.index = a.result(method= 'features')
    # df.reset_index(level=0, inplace=True)
    # df.columns =["Feature", "Value"]
    # pandas.set_option('display.float_format', lambda x: '%.3f' % x)
    # return df

    FeaturesList = [('Feature', 'Value')]

    for i in xrange(len(a.result(method='array'))):

        FeaturesList.append((a.result(method='features')[i],
                             a.result(method='array')[i]))

    a = make_table(FeaturesList)
    apply_theme('basic')
    set_global_style(float_format='%0.5f')

    return a
