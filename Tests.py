import AFTest
import CFTest
import LayersTest
import DNTest
import numpy as np


def test_suite():

    np.random.seed(10)
    res = list()
    res.append(AFTest.test_suite())
    res.append(CFTest.test_suite())
    res.append(LayersTest.test_suite())
    res.append(DNTest.test_suite())

    if np.array(res).all():
        print('ALL TESTS RAN SUCCESSFULLY')


if __name__ == '__main__':
    test_suite()
