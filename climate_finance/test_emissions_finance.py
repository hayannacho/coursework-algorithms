import unittest, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from emissions_finance import *


class Tests(unittest.TestCase):
    def test_values(self):
        y, d = analyze(*synthetic_data())
        self.assertEqual(y, 2025)
        self.assertEqual(len(d), 6)
        self.assertAlmostEqual(d.coal_share.iloc[0], 0.1)

    def test_alias_and_numeric(self):
        d = standardize_country_names(pd.DataFrame({"country": ["Republic of Korea"]}), "country")
        self.assertEqual(d.country[0], "South Korea")
        self.assertEqual(coerce_numeric_series(pd.Series(["1,000"]))[0], 1000)

    def test_duplicates(self):
        e, f = synthetic_data()
        with self.assertRaises(ValueError):
            analyze(e, pd.concat([f, f]))


if __name__ == "__main__":
    unittest.main()
