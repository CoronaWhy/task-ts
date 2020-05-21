import unittest
from corona_ts.data_utils.data_crawler import load_data
class TestCoreMethods(unittest.TestCase):

    def test_upper(self):
        self.assertTrue(len(load_data()))

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

if __name__ == '__main__':
    unittest.main()
