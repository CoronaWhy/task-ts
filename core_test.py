import unittest
from corona_ts.data_utils.data_crawler import load_data
from corona_ts.data_utils.data_creator import loop_through_locations, region_df_format
class TestCoreMethods(unittest.TestCase):

    def test_load_data(self):
        self.assertTrue(len(load_data()))

    def test_region(self):
        df = load_data()
        df_ny = region_df_format(df, 'New York_New York County')
        self.assertTrue(df_ny)
        self.assertEqual(df_ny.iloc[0]['new_cases'], 0)
        self.assertEqual(df_ny.iloc[0]['deaths'], 0)
    
if __name__ == '__main__':
    unittest.main()
