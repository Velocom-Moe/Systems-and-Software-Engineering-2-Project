import data
import pandas as pd

# Load all files from csv and save them into a pickle. If a pickle is already existing
# use this function instead: data.loadPickle()
bookings = data.loadFilter(cityFilter='Frankfurt am Main')
stations = data.loadStation()

# Delet NaN-Values
bookings = bookings.dropna()

# This method call finally creates our needed framework and data files for further working.
dataMatrix = data.createMatrix(bookings, stations)
