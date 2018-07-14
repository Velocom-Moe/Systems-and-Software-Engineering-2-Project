"""
Author: Philipp
Version: 1.3

This class provides functions to collect necessary data.
For this create a folder called Opendata Bahn and save the data within.
"""

import pandas as pd
import pickle as pkl


def loadPickle(path='Opendata Bahn/frankfurt.pkl'):
    """This function load a saved pickle file into a pandas data frame.
    The path is for location and name of the file.

    Keyword Arguments:
        path {str} -- [description] (default: {'Opendata Bahn/frankfurt.pkl'})

    Returns:
        dataframe {pandas data frame} -- Loaded pickle file saved in a pandas data frame
    """
    f = open(path, 'rb')
    dataframe = pkl.load(f)
    print('Loaded {}.pkl from hdd'.format(path))
    return dataframe


def dataLoader(cityFilter=None, number=-1):
    """Function for loading CSV-data and filter them by city. The data will be 
    loaded in chunks because of the size of the data. At the end they will be merged 
    again.

    Keyword Arguments:
        cityFilter {str, None} --  Set the city filter. (default: {None})
        number {int} -- Number of rows collecting data. Default is set to laod all(default: -1)

    Returns:
        data {pandas data frame} -- The data frame containing the data.
    """
    if (number > 0):
        loadChunk = pd.read_csv('Opendata Bahn/OPENDATA_BOOKING_CALL_A_BIKE.csv', nrows=number,
                                index_col='DATE_BOOKING', delimiter=';', parse_dates=['DATE_BOOKING'], chunksize=1000)
        load = pd.concat(list(loadChunk))
    else:
        loadChunk = pd.read_csv('Opendata Bahn/OPENDATA_BOOKING_CALL_A_BIKE.csv',
                                index_col='DATE_BOOKING', delimiter=';', parse_dates=['DATE_BOOKING'], chunksize=1000)
        load = pd.concat(list(loadChunk))
    if (isinstance(cityFilter, str)):
        data = load[load['CITY_RENTAL_ZONE'] == cityFilter]
        data = data[['BOOKING_HAL_ID', 'VEHICLE_HAL_ID', 'CUSTOMER_HAL_ID', 'DATE_FROM', 'DATE_UNTIL', 'DISTANCE',
                     'START_RENTAL_ZONE', 'START_RENTAL_ZONE_HAL_ID', 'END_RENTAL_ZONE', 'END_RENTAL_ZONE_HAL_ID', 'CITY_RENTAL_ZONE']]
    else:
        data = load[['BOOKING_HAL_ID', 'VEHICLE_HAL_ID', 'CUSTOMER_HAL_ID', 'DATE_FROM', 'DATE_UNTIL', 'DISTANCE',
                     'START_RENTAL_ZONE', 'START_RENTAL_ZONE_HAL_ID', 'END_RENTAL_ZONE', 'END_RENTAL_ZONE_HAL_ID', 'CITY_RENTAL_ZONE']]
    return data


def loadFilter(year=None, cityFilter=None, number=-1):
    """Extends the dataLoader function with additional filter. Aditional there 
    are functions implemented for a year separating collecting of the data.

    Keyword Arguments:
        year {int, None} -- Integer for filtering the data by year (default: {None})
        cityFilter {str, None} --  Set the city filter. (default: {None})
        number {int} -- Number of rows collecting data. (default: -1)

    Returns:
        data frame -- The pandas data frame containing the data.
    """

    if (isinstance(year, str)):
        path = '{}.pkl'.format(year)
        try:
            f = open(path, 'rb')
            dataframe = pkl.load(f)
            print('Loaded {}.pkl from hdd'.format(year))
            return dataframe[dataframe['CITY_RENTAL_ZONE'] == cityFilter]
        except (OSError, IOError, FileNotFoundError):
            print("Import data from csv. Caution: This takes a wile.")
            if(number > 0):
                tmp = dataLoader(cityFilter, number)
            else:
                tmp = dataLoader(cityFilter)
            tmp['2014'].to_pickle("./2014.pkl")
            tmp['2015'].to_pickle("./2015.pkl")
            tmp['2016'].to_pickle("./2016.pkl")
            tmp['2017'].to_pickle("./2017.pkl")
            tmp.to_pickle("./total.pkl")
            print("Data have been loaded and splittet by year.")
            return tmp[year]
    else:
        print("Load all data into data frame.")
        if(number > 0):
            tmp = dataLoader(cityFilter, number)
        else:
            tmp = dataLoader(cityFilter)
        print("Loaded all files. Return now:")
        return tmp


def loadStation(path='Opendata Bahn/OPENDATA_RENTAL_ZONE_CALL_A_BIKE.csv'):
    """This function takes the station data csv file sort them by Frankfurt am Main 
    and load them into a data frame.

    Keyword Arguments:
        path {str} --  (default: {'Opendata Bahn/OPENDATA_RENTAL_ZONE_CALL_A_BIKE.csv'})

    Returns:
        result {pandas data frame} -- Loaded file saved in a pandas data frame.
    """
    data = pd.read_csv(path, delimiter=';')
    result = data[data['CITY'] == 'Frankfurt am Main']
    return result


def createMatrix(bookings, stations):
    """Method for creating and filling the data frame for further working.
    The Index is date time and the columns are the station ids. The number
    within are the total drive in bikes to this station.
    Last Row will be 'total' which represents the total amount of driven
    bikes a day (sum of the row)

    Arguments:
        booking {data frame} -- Loaded booking data in a pandas data frame
        stations {data frame} -- Loaded station information data in a pandas data frame
        stationsListFfmTotal.csv -- Save the data frame into this file.
        stationsListFfmTotal.pkl -- Save the data frame into this file
    """
    # creating raw matrix with zeros and needed structure
    dataIndex = pd.date_range('1/1/2014', periods=1232, freq='D')
    feature_list = []
    for station in stations['RENTAL_ZONE_HAL_ID']:
        feature_list.append(station)
    dataFrame = pd.DataFrame(0, index=dataIndex, columns=feature_list)

    # Function for fill in data into
    emptyStations = {}
    steps = 0
    for index, row in bookings.iterrows():
        steps += 1
        rentalZoneID = int(row['START_RENTAL_ZONE_HAL_ID'])
        bookingDay = index
        bookingDay = bookingDay.replace(hour=0, minute=0, second=0)
        try:
            dataFrame.at[bookingDay, rentalZoneID] += 1
        except (KeyError):
            emptyStations[(rentalZoneID)] = 1
            print(rentalZoneID)
        if(steps % 10000 == 0):
            print(str(steps))
    print("done")

    # Creating resulting data frame with total column
    total = pd.DataFrame(dataFrame.sum(axis=1))
    total.to_csv('Opendata Bahn/stationsListFfmTotal.csv')
    total.to_pickle('Opendata Bahn/stationsListFfmTotal.pkl')
    dataFrame['total'] = pd.DataFrame(total)

    return dataFrame
