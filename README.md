# Developing a model to predict the prices for Airbnb Hotels

## Dataset Information
Fields

Id, Property_type, Room_type, Ammenities, Accomodates, bathroom_number, bedroom_number, bed_type, cancellation_policy, cleaning_fee, City, Description, First_review, Latitude, Longitude, Name, neighbourhood, Number of Reviews, Review_score_rating

Variable to be predicted - Log_price

## Functions for Data Preprocessing

1. For handling missing Data - MissingDataHandling.py
2. Creation of New Columns - CombiningColumns.py
3. Adding Sentiment Score after analyzing the Review Comments - Add_Sentiments.py

## Random Forest Model is used as the ML model

Code in - AirBnB/sample_script.py
