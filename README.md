# Software-Engineering-2

This repository is dedicated to publish the created code during the software engineering 2 lecture in summer term 2018. 
All work is done with python 3.6 and the newest version of the used libraries.

## Data
The used data are from [Open Date Portal of Deutsche Bahn](http://data.deutschebahn.com/dataset/data-call-a-bike). 
The downloaded and extracted files has to be put and extracted into the 'Opendata Bahn' folder. 
The size is to big for uploading them to this Repository.
Following Files need to be downloaded:

- Stationen Call a Bike (Stand 05/2017)  Filename: OPENDATA_RENTAL_ZONE_CALL_A_BIKE.csv
- Buchungen Call a Bike (Stand 05/2017)  Filename: OPENDATA_BOOKING_CALL_A_BIKE.csv


## Data loading
If you want to start the presented code you have to extract the data in a needed format and data structure. 
You can do this by running dataStation.py or do an own file with this codes:

```python
> import data
> import pandas as pd

> # Load all files from csv and save them into a pickle. If a pickle is already existing
> # use this function instead: data.loadPickle()
> bookings = data.loadFilter(cityFilter='Frankfurt am Main')
> stations = data.loadStation()

> # Delet NaN-Values
> bookings = bookings.dropna()

> # This method call finally creates our needed framework and data files for further working.
> dataMatrix = data.createMatrix(bookings, stations)

```

Important: You need to have the folder 'Opendata Bahn' within your directory or change the path in the corresponding function calls.


## Keras
Run the hole nn-keras.py file to get the needed results.

## MXNet
Run the hole nn-mxnet.py file to get the needed results. 

## Theano
Run the hole rnn.py file from the 'Theano' folder to get the needed results.
