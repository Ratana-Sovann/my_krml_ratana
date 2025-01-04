import unittest
import pandas as pd
from my_krml_ratana.features.plot import histogram_boxplot, labeled_barplot, plot_cities


class TestPlotFunctions(unittest.TestCase):

    def setUp(self):
        """Set up sample data for testing."""
        self.data = pd.DataFrame({
            'numerical': [1, 2, 2, 3, 3, 3, 4, 4, 4, 4],
            'categorical': ['A', 'B', 'A', 'B', 'C', 'A', 'C', 'B', 'C', 'A'],
            'lat': [34.05, 36.16, 40.71, 37.77, 34.05, 36.16, 40.71, 37.77, 34.05, 36.16],
            'long': [-118.24, -115.15, -74.00, -122.41, -118.24, -115.15, -74.00, -122.41, -118.24, -115.15],
            'city': ['CityA', 'CityB', 'CityC', 'CityD', 'CityA', 'CityB', 'CityC', 'CityD', 'CityA', 'CityB']
        })

    def test_histogram_boxplot(self):
        """Test the histogram_boxplot function."""
        try:
            histogram_boxplot(data=self.data, feature='numerical', figsize=(10, 6), kde=True, bins=5)
        except Exception as e:
            self.fail(f"histogram_boxplot raised an exception: {e}")

    def test_labeled_barplot_counts(self):
        """Test the labeled_barplot function with count labels."""
        try:
            labeled_barplot(data=self.data, feature='categorical', perc=False, n=2)
        except Exception as e:
            self.fail(f"labeled_barplot with counts raised an exception: {e}")

    def test_labeled_barplot_percentages(self):
        """Test the labeled_barplot function with percentage labels."""
        try:
            labeled_barplot(data=self.data, feature='categorical', perc=True, n=3)
        except Exception as e:
            self.fail(f"labeled_barplot with percentages raised an exception: {e}")

    def test_plot_cities_interactive(self):
        """Test the plot_cities function in interactive mode."""
        try:
            city_map = plot_cities(df=self.data, lat='lat', long='long', city='city', interactive=True)
            self.assertIsNotNone(city_map, "plot_cities should return a Folium map in interactive mode.")
            self.assertEqual(city_map.__class__.__name__, 'Map', "The returned object should be a Folium Map.")
        except Exception as e:
            self.fail(f"plot_cities in interactive mode raised an exception: {e}")

    def test_plot_cities_static(self):
        """Test the plot_cities function in static mode."""
        try:
            plot_cities(df=self.data, lat='lat', long='long', city=None, interactive=False)
        except Exception as e:
            self.fail(f"plot_cities in static mode raised an exception: {e}")


if __name__ == '__main__':
    unittest.main()
