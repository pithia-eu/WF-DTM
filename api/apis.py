from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os
from datetime import datetime, timedelta

AP_TO_KP = {
    0:0,
    2:0.33,
    3:0.66,
    4:1,
    5:1.33,
    6:1.66,
    7:2,
    9:2.33,
    12:2.66,
    15:3,
    18:3.33,
    22:3.66,
    27:4,
    32:4.33,
    39:4.66,
    48:5,
    56:5.33,
    67:5.66,
    80:6,
    94:6.33,
    111:6.66,
    132:7,
    154:7.33,
    179:7.66,
    207:8,
    236:8.33,
    300:8.66,
    400:9,
}

# Get the full path to the directory containing the FastAPI script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the full path to the directory containing the NOA Workflow scripts
workflow_dir = script_dir.replace('/api', '')
app = FastAPI(
    openapi_tags=[
        {
            "name": "Run Workflow",
            "description": "Run the DTM2020 Workflow"
        },
    ],
    title="DTM2020 Workflow API",
    description="The DTM2020 workflow includes a total of 16 runs on the DTM2020-operational: semi-empirical thermosphere model.",
    version="1.0.0",
)

# Configure CORS for all domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Define the new `run_workflow` API, accept two query parameters: `date` (from 1970-01-01 to yesterday) and `altitute` (from 120 to 1500 km)
@app.get("/run_workflow/", response_class=StreamingResponse, responses={200: {"content": {"application/octet-stream": {}},"description": "**Important:** Please remember to rename the downloaded file to have the extension '*.zip' before opening it.\n\n",}},summary="Run the DTM2020 Workflow", description="Return the 16 runs parameters for the DTM2020 Model in JSON format.\n\n"+"**Important:** When selecting the 'zip' format, please remember to rename the downloaded file to have the extension '*.zip' before opening it.\n\n", tags=["Run Workflow"])
async def run_workflow(date: str = Query(..., description="Date in the format 'YYYY-MM-DD', e.g. 2024-01-01. The date should be from 1970-01-01"), altitude: int = Query(120, ge=120, le=1500, description="Altitude in km, from 120 to 1500 km.")):
    # Validate the date
    try:
        datetime.strptime(date, '%Y-%m-%d')
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Ensure the format is YYYY-MM-DD.")
    # Validate the date range from 1970-01-01 to yesterday
    yesterday = datetime.now() - timedelta(days=1)
    if datetime.strptime(date, '%Y-%m-%d') > yesterday or datetime.strptime(date, '%Y-%m-%d') < datetime(1970, 1, 1):
        raise HTTPException(status_code=400, detail="The date should be from 1970-01-01 to yesterday.")

    # Construct the start date and end date for the KP data, start date is previous 81 days from the date, end date is the day after the date
    start_date = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=80)).strftime('%Y-%m-%d')
    end_date = (datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    select_date = datetime.strptime(date, '%Y-%m-%d')
    previous_date = select_date - timedelta(days=1)
    print(f"Start date: {start_date}, End date: {end_date}")
    
    # The URL of the dataset
    DATA_URL="https://kp.gfz-potsdam.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt"
    # With new date index from Year, Month, Day, skip the first 40 rows, and set the column names
    df = pd.read_csv(DATA_URL, skiprows=40, sep='\s+', names=["Year", "Month", "Day", "Days", "Days_M", "Bsr", "dB",
                        "Kp1", "Kp2", "Kp3", "Kp4", "Kp5", "Kp6", "Kp7", "Kp8",
                        "ap1", "ap2", "ap3", "ap4", "ap5", "ap6", "ap7", "ap8",
                        "Ap", "SN", "F10.7obs", "F10.7adj", "D"])
    # Add the date index to the dataframe
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    # Filter the dataset from the start_date to end_date, include end_date, by comparing Date index
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    # Get the mean value of F10.7obs for two date range, 1. from start_date to (end_date - 1), 2. from (start_date + 1) to end_date
    mean_f107_1 = df[(df['Date'] >= start_date) & (df['Date'] < end_date)]['F10.7obs'].mean()
    mean_f107_2 = df[(df['Date'] > start_date) & (df['Date'] <= end_date)]['F10.7obs'].mean()
    print(f"81-day Mean F10.7obs for {select_date}: {mean_f107_1}")
    print(f"81-day Mean F10.7obs for {end_date}: {mean_f107_2}")
    # Get the previous day and current day 'Ap' value
    ap_1 = df[df['Date'] == previous_date]['Ap'].values[0]
    ap_2 = df[df['Date'] == select_date]['Ap'].values[0]
    # Find the closest of AP_TO_KP dictionary key to the 'Ap' value, and get the corresponding Kp value
    kp_1 = AP_TO_KP[min(AP_TO_KP.keys(), key=lambda x:abs(x-ap_1))]
    kp_2 = AP_TO_KP[min(AP_TO_KP.keys(), key=lambda x:abs(x-ap_2))]
    print(f"Kp value for {previous_date}: {ap_1} -> {kp_1}")
    print(f"Kp value for {select_date}: {ap_2} -> {kp_2}")
    # Get the F10.7obs value for the previous day and current day
    f107_1 = df[df['Date'] == previous_date]['F10.7obs'].values[0]
    f107_2 = df[df['Date'] == select_date]['F10.7obs'].values[0]
    print(f"F10.7obs value for {previous_date}: {f107_1}")
    print(f"F10.7obs value for {select_date}: {f107_2}")
    # Get the number of days for current day and end_date from the start of year
    days_1 = (select_date - datetime(select_date.year, 1, 1)).days + 1
    days_2 = (datetime.strptime(end_date, '%Y-%m-%d') - datetime(datetime.strptime(end_date, '%Y-%m-%d').year, 1, 1)).days + 1
    print(f"Number of days from the start of the year for {select_date}: {days_1}")
    print(f"Number of days from the start of the year for {end_date}: {days_2}")
    # For days_1, create an array to store the 'Kp8' value from the previou_day, and the 'Kp1' to 'Kp7' value from the current day
    Kp_pre_1 = df[df['Date'] == previous_date][['Kp8']].values[0]
    Kps_1 = df[df['Date'] == select_date][['Kp1', 'Kp2', 'Kp3', 'Kp4', 'Kp5', 'Kp6', 'Kp7']].values[0].tolist()
    # Add the Kp_pre_1 value to the Kps_1 array, at the beginning
    Kps_1.insert(0, Kp_pre_1[0])
    # For days_2, create an array to store the 'Kp8' value from the current day, and the 'Kp1' to 'Kp7' value from the end day
    Kp_pre_2 = df[df['Date'] == select_date][['Kp8']].values[0]
    Kps_2 = df[df['Date'] == datetime.strptime(end_date, '%Y-%m-%d')][['Kp1', 'Kp2', 'Kp3', 'Kp4', 'Kp5', 'Kp6', 'Kp7']].values[0].tolist()
    # Add the Kp_pre_2 value to the Kps_2 array, at the beginning
    Kps_2.insert(0, Kp_pre_2[0])
    print(f"Kp values for {select_date}: {Kps_1}")
    print(f"Kp values for {end_date}: {Kps_2}")
    output_json = {
        "date": date,
        "altitude": altitude,
        'runs':[
            {'day':days_1,
             'fm':mean_f107_1,
             'fl':f107_1,
             'alt':altitude,
             'akp1':Kps_1,
             'akp3':kp_1,
            },
            {'day':days_2,
             'fm':mean_f107_2,
             'fl':f107_2,
             'alt':altitude,
             'akp1':Kps_2,
             'akp3':kp_2,
            }
        ]
    }
    return JSONResponse(status_code=200, content=output_json)