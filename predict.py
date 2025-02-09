import sys
import pandas as pd
import numpy as np
import pickle
import json

# Load the model
model_file_path = r'F:\FPML\rf_random.pkl'
with open(model_file_path, 'rb') as file:
    model = pickle.load(file)

# Define preprocessing function
def preprocess_input(input_data):
    """
    Preprocess the input data to match the model's training data structure.
    """
    # Convert input data into DataFrame
    df = pd.DataFrame([input_data])

    # Perform all necessary preprocessing steps
    # Map 'Total_Stops' column
    stops_mapping = {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}
    df['Total_Stops'] = df['Total_Stops'].map(stops_mapping)

    air = ['Trujet', 'SpiceJet', 'Air Asia', 'IndiGo', 'GoAir', 'Vistara',
       'Vistara Premium economy', 'Air India', 'Multiple carriers',
       'Multiple carriers Premium economy', 'Jet Airways',
       'Jet Airways Business']
    dict1 = {key:index for index,key in enumerate(air,0)}
    df['Airline'] = df['Airline'].map(dict1)
    if df['Airline'].isnull().any():
        df['Airline'] = df['Airline'].fillna(0).astype(int)  # Replace NaN with -1 or another default value


    # One-hot encode 'Source'
    for category in ['Delhi', 'Mumbai', 'Chennai', 'Kolkata', 'Bangalore']:
        df[f'Source_{category}'] = df['Source'].apply(lambda x: 1 if x == category else 0)

    # One-hot encode 'Destination'
    for category in ['Delhi', 'Mumbai', 'Chennai', 'Kolkata', 'Bangalore']:
        df[f'Destination_{category}'] = df['Destination'].apply(lambda x: 1 if x == category else 0)
    
    dest = ['Kolkata', 'Hyderabad', 'Delhi', 'Banglore', 'Cochin']
    dict2 = {key:index for index,key in enumerate(dest,0)}
    df['Destination'] = df['Destination'].map(dict2)
    if df['Destination'].isnull().any():
        df['Destination'] = df['Destination'].fillna(0).astype(int)  # Replace NaN with -1 or another default value


    # Add missing features and ensure all expected columns are present
    df['journey_day'] = df['Journey_Day']  # Example default value, you can adjust
    df['journey_month'] = df['Journey_Month']  # Example default value, you can adjust

    # Extract Departure Time (hour and minute)
    if 'Dep_Time' in df:
        dep_time = df['Dep_Time'].iloc[0]  # Get the first row value as a string
        dep_time_hr, dep_time_min = map(int, dep_time.split(':'))
        df['Dep_Time_hr'] = dep_time_hr
        df['Dep_Time_min'] = dep_time_min
        df.drop(columns=['Dep_Time'], inplace=True)

    # Extract Arrival Time (hour and minute)
    if 'Arrival_Time' in df:
        arr_time = df['Arrival_Time'].iloc[0]  # Get the first row value as a string
        arr_time_hr, arr_time_min = map(int, arr_time.split(':'))
        df['Arrival_Time_hr'] = arr_time_hr
        df['Arrival_Time_min'] = arr_time_min
        df.drop(columns=['Arrival_Time'], inplace=True)

    # Extract Duration (hour and minute)
    if 'Duration' in df:
        duration = df['Duration'].iloc[0]  # Get the first row value as a string
        duration_parts = duration.split(' ')
        duration_hr = int(duration_parts[0].replace('h', '')) if 'h' in duration_parts[0] else 0
        duration_min = int(duration_parts[1].replace('m', '')) if len(duration_parts) > 1 else 0
        df['Duration_hr'] = duration_hr
        df['Duration_mins'] = duration_min
        df.drop(columns=['Duration'], inplace=True)

    # Drop unused columns
    df.drop(columns=['Source', 'Route', 'Additional_Info'], inplace=True)

    # Reorder columns to match model training
    model_columns = [
        'Airline', 'Total_Stops', 'journey_day', 'journey_month', 
        'Dep_Time_hr', 'Dep_Time_min', 'Arrival_Time_hr', 'Arrival_Time_min',
        'Duration_hr', 'Duration_mins',
        'Source_Delhi', 'Source_Mumbai', 'Source_Chennai', 'Source_Kolkata', 'Source_Bangalore',
        'Destination'
    ]

    # Add missing columns with default values (0) if not present
    for col in model_columns:
        if col not in df:
            df[col] = 0

    # Reorder columns to match the model's training data
    df = df[model_columns]

    # print("Processed DataFrame:\n", df)
    return df


# Accept input data as JSON
try:
    input_data = json.loads(sys.argv[1])
    
    # print("Input Data Received:\n", input_data)

    # Preprocess the input data
    processed_data = preprocess_input(input_data)

    try:
    # Assuming `prediction` is your result
        prediction = model.predict(processed_data)
        result = {"Prediction": prediction.tolist()}  # Convert NumPy array to list if needed
        print(json.dumps(result))  # Output as JSON
    except Exception as e:
        error_message = {"error": str(e)}
        print(json.dumps(error_message))  # Send error as JSON
    
    # # Make prediction
    # prediction = model.predict(processed_data)
    # print("Prediction:", prediction[0])

except Exception as e:
    print("Error occurred:", e)
