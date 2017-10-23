# User_Location_Density_Categorization
Written in Python for Spark for user categorisation based on location (latitude and longitude)

This project is to address below real life problem. Please note that data given is jst sample. Real data cannot be shared. Real data is also large in volume.

Problem Statement:
An organization wants to do campaign for different kinds of customers. They like to study density of customer for different location based on customer complaint received from various location. Geo-location is based on latitude and longitude. In this problem we are own for clustering customers into three classes depending on the geo-location of those customers.

Data file:
Sample data file contains several fields comma separated. We are interested only last two fields which represent latitude and longitude. rest of the fields are simple ignored

Solution:
We have applied K-Means machine learning algorithm here. The program is written in python and executed in Spark platform. Therefore, several required Spark libraries are imported. 

Preprocess:
Data is real life. Hence, preprocessing and cleanup are required. There are entries where latitude = 0 and longitude = 0. However, in practical life it is not real that any customer stays in that kind of location. So, those entries are cleaned before processing.

Prerequisite:
1) You need to have Python installed version 3.5 or above and environment setup properly
2) You need to have Spark installed and environment variables setup properly

Usage:
It is written and tested for Windows 7 OS. It should run in other environment also, but not tested. May be little bit changes are required as per demand of that OS. Please keep data file in same folder where source program is located. Data file is without any header.
From command line following command should be executed.

spark-submit clusterCustomer.py


Output:
It will generate a comma separated file, "sample_output.csv" in same folder where datafile and source program are located. Output file will have three fields separated with comma. Fields are as below
1) latitude
2) longitude
3) cluster label (0, 1 or 2)



************  Enjoy and don't forget to give credit if it helps ****************
