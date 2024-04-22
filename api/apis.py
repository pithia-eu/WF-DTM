import json
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os
from datetime import datetime, timedelta
import shutil

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
    description="<p>The DTM2020 model predictions are calculated at a specified altitude (latitude - local time grids) from the specified start date every 3 hours for 3 days (i.e. 24 epochs, 0h, +3h, +6h, etc) using the observed F10.7 and geomagnetic activity indices Kp.</p><br/><p>The effect of a geomagnetic storm can be visualized effectively with this workflow.</p><br/><p>The partial density and temperature grids and plots, and the observed F10.7 flux and Kp, can be downloaded by clicking on 'Download file' (rename the file and add .zip as an extension).</p>",
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
    # Validate the date range from 1970-01-01 to 3 days ago
    days_3_ago = datetime.now() - timedelta(days=3)
    if datetime.strptime(date, '%Y-%m-%d') > days_3_ago or datetime.strptime(date, '%Y-%m-%d') < datetime(1970, 1, 1):
        raise HTTPException(status_code=400, detail="The date should be from 1970-01-01 to 3 days ago.")

    final_zip_file = f"{script_dir}/output/{date}/{altitude}/final_output.zip"
    # Check whether the final zip file is already created
    if os.path.exists(final_zip_file):
        return FileResponse(final_zip_file, media_type='application/octet-stream', filename='final_output.zip')
    
    # Create the final output folder
    final_output_folder = f"{script_dir}/output/{date}/{altitude}/final"
    # If the final output folder exists, remove it
    if os.path.exists(final_output_folder):
        os.system(f"rm -rf {final_output_folder}")
    else:
        # Create the final output folder if it does not exist
        os.makedirs(final_output_folder, exist_ok=True)
        
    
    # Construct the start date and end date for the KP data, start date is previous 81 days from the date, end date is 2 days after the date
    start_date = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=80)).strftime('%Y-%m-%d')
    start_1_date = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=79)).strftime('%Y-%m-%d')
    end_date = (datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    end_2_date = (datetime.strptime(date, '%Y-%m-%d') + timedelta(days=2)).strftime('%Y-%m-%d')
    select_date = datetime.strptime(date, '%Y-%m-%d')
    previous_date = select_date - timedelta(days=1)
    print(f"Start date: {start_date}, End date: {end_2_date}")
    
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
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_2_date)]
    # Get the mean value of F10.7obs for two date range, 1. from start_date to (end_date - 1), 2. from (start_date + 1) to end_date, 3. from start_1_date to (end_2_date - 1)
    mean_f107_1 = df[(df['Date'] >= start_date) & (df['Date'] <= select_date)]['F10.7obs'].mean()
    mean_f107_2 = df[(df['Date'] > start_date) & (df['Date'] <= end_date)]['F10.7obs'].mean()
    mean_f107_3 = df[(df['Date'] > start_1_date) & (df['Date'] <= end_2_date)]['F10.7obs'].mean()
    # convert df[(df['Date'] >= start_date) & (df['Date'] <= select_date)]['F10.7obs'] to JSON, with ["Date" format as "YYYY-MM-DD", "F10.7obs"]
    f107_1_json = df[(df['Date'] >= start_date) & (df['Date'] <= select_date)][['Date', 'F10.7obs']].to_json(orient='records')
    f107_2_json = df[(df['Date'] > start_date) & (df['Date'] <= end_date)][['Date', 'F10.7obs']].to_json(orient='records')
    f107_3_json = df[(df['Date'] > start_1_date) & (df['Date'] <= end_2_date)][['Date', 'F10.7obs']].to_json(orient='records')
    # Load the JSON string to JSON object, and convert the date (epoch time) to date string format
    f107_1_json = json.loads(f107_1_json)
    for item in f107_1_json:
        item['Date'] = datetime.fromtimestamp(item['Date']/1000).strftime('%Y-%m-%d')
    f107_2_json = json.loads(f107_2_json)
    for item in f107_2_json:
        item['Date'] = datetime.fromtimestamp(item['Date']/1000).strftime('%Y-%m-%d')
    f107_3_json = json.loads(f107_3_json)
    for item in f107_3_json:
        item['Date'] = datetime.fromtimestamp(item['Date']/1000).strftime('%Y-%m-%d')
        
    print(f"81-day Mean F10.7obs for {select_date}: {mean_f107_1}")
    print(f"81-day Mean F10.7obs for {end_date}: {mean_f107_2}")
    print(f"81-day Mean F10.7obs for {end_2_date}: {mean_f107_3}")
    # Get the previous day and current day 'Ap' value
    ap_1 = df[df['Date'] == previous_date]['Ap'].values[0]
    ap_2 = df[df['Date'] == select_date]['Ap'].values[0]
    ap_3 = df[df['Date'] == end_date]['Ap'].values[0]
    # Find the closest of AP_TO_KP dictionary key to the 'Ap' value, and get the corresponding Kp value
    kp_1 = AP_TO_KP[min(AP_TO_KP.keys(), key=lambda x:abs(x-ap_1))]
    kp_2 = AP_TO_KP[min(AP_TO_KP.keys(), key=lambda x:abs(x-ap_2))]
    kp_3 = AP_TO_KP[min(AP_TO_KP.keys(), key=lambda x:abs(x-ap_3))]
    print(f"Kp value for {previous_date}: {ap_1} -> {kp_1}")
    print(f"Kp value for {select_date}: {ap_2} -> {kp_2}")
    print(f"Kp value for {end_date}: {ap_3} -> {kp_3}")
    # Get the F10.7obs value for the previous day and current day
    f107_1 = df[df['Date'] == previous_date]['F10.7obs'].values[0]
    f107_2 = df[df['Date'] == select_date]['F10.7obs'].values[0]
    f107_3 = df[df['Date'] == end_date]['F10.7obs'].values[0]
    print(f"F10.7obs value for {previous_date}: {f107_1}")
    print(f"F10.7obs value for {select_date}: {f107_2}")
    print(f"F10.7obs value for {end_date}: {f107_3}")
    # Get the number of days for current day and end_date from the start of year
    days_1 = (select_date - datetime(select_date.year, 1, 1)).days + 1
    days_2 = (datetime.strptime(end_date, '%Y-%m-%d') - datetime(datetime.strptime(end_date, '%Y-%m-%d').year, 1, 1)).days + 1
    days_3 = (datetime.strptime(end_2_date, '%Y-%m-%d') - datetime(datetime.strptime(end_2_date, '%Y-%m-%d').year, 1, 1)).days + 1
    print(f"Number of days from the start of the year for {select_date}: {days_1}")
    print(f"Number of days from the start of the year for {end_date}: {days_2}")
    print(f"Number of days from the start of the year for {end_2_date}: {days_3}")
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
    Kp_pre_3 = df[df['Date'] == end_date][['Kp8']].values[0]
    Kps_3 = df[df['Date'] == datetime.strptime(end_2_date, '%Y-%m-%d')][['Kp1', 'Kp2', 'Kp3', 'Kp4', 'Kp5', 'Kp6', 'Kp7']].values[0].tolist()
    # Add the Kp_pre_3 value to the Kps_3 array, at the beginning
    Kps_3.insert(0, Kp_pre_3[0])
    print(f"Kp values for {select_date}: {Kps_1}")
    print(f"Kp values for {end_date}: {Kps_2}")
    print(f"Kp values for {end_2_date}: {Kps_3}")
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
             'f107':f107_1_json,
             'ap':ap_1,
            },
            {'day':days_2,
             'fm':mean_f107_2,
             'fl':f107_2,
             'alt':altitude,
             'akp1':Kps_2,
             'akp3':kp_2,
             'f107':f107_2_json,
             'ap':ap_2,
            },
            {'day':days_3,
             'fm':mean_f107_3,
             'fl':f107_3,
             'alt':altitude,
             'akp1':Kps_3,
             'akp3':kp_3,
             'f107':f107_3_json,
             'ap':ap_3,
             }
            
        ]
    }
    # For each runs, print the parameters
    run_responses = []
    try:
        for run in output_json['runs']:
            hour = 0 # Hour, every 3 hours
            # For each akp1 value, print the akp1 value
            for akp1 in run['akp1']:
                # the input parameter for the DTM2020 model is (fm, fl, alt, day, akp1, akp3)
                run_params = (run['fm'], run['fl'], run['alt'], run['day'], akp1, run['akp3'])
                # Construct the request URL for the DTM2020 model: https://dtm.pithia.eu/execute?fm=180&fl=100&alt=300&day=180&akp1=0&akp3=0
                run_request = f"https://dtm.pithia.eu/execute?fm={run_params[0]}&fl={run_params[1]}&alt={run_params[2]}&day={run_params[3]}&akp1={akp1}&akp3={run_params[5]}"
                # Run the request URL
                run_response = requests.get(run_request)
                # Response: [{"execution_id": xxx}], change the execution id to day_hour
                run_response_json = run_response.json()
                new_execution_id = f"{run_params[3]}_{hour}"
                # append the run response to the run_responses array
                run_responses.append({new_execution_id: run_response_json[0]["execution_id"]})
                hour += 3
        for execution in run_responses:
            for key, value in execution.items():
                # Construct the request URL to download the results: https://dtm.pithia.eu/results?execution_id=xxx
                results_request = f"https://dtm.pithia.eu/results?execution_id={value}"
                # print the response content file name and size
                # print(f"File name: {results_response.headers['Content-Disposition']}, File size: {results_response.headers['Content-Length']}, Request Execution: {key} {value}")
                # The response content is a zip file, it contains the following files: 'DTM20F107Kp_N2.datx', 'DTM20F107Kp_N2.png', 'DTM20F107Kp_ro.datx', 'DTM20F107Kp_ro.png' ...
                # Need to extract and rename all the .datx and .png files by replacing the 'DTM20F107Kp' with the key
                # Step 1: Save the zip file, to /output/date/altitude/ folder, create the folder if not exist
                output_folder = f"{script_dir}/output/{date}/{altitude}"
                # Create the output folder if not exist
                os.makedirs(output_folder, exist_ok=True)
                # Check whether the file is already downloaded
                if os.path.exists(f"{output_folder}/{key}.zip"):
                    print(f"The file {key}.zip is already downloaded.")
                else:
                    # Run the request URL
                    results_response = requests.get(results_request)
                    # Save the zip file to the output folder
                    with open(f"{output_folder}/{key}.zip", 'wb') as f:
                        f.write(results_response.content)
                        # Unzip the file
                # Check whether the file is already unzipped
                if os.path.exists(f"{output_folder}/{key}"):
                    print(f"The file {key} is already unzipped.")
                else:
                    os.system(f"unzip {output_folder}/{key}.zip -d {output_folder}/{key}")
    
        folder_metrics = ['He','N2','O','ro','Tinf','Tz']
        # For each folder metric, create the folder, plots_metric and datas_metric
        for folder_metric in folder_metrics:
            os.makedirs(f"{final_output_folder}/plots_{folder_metric}/", exist_ok=True)
            os.makedirs(f"{final_output_folder}/datas_{folder_metric}/", exist_ok=True)
            
        for execution in run_responses:
            for key, value in execution.items():
                # Locate the unzipped folder
                unzipped_folder = f"{output_folder}/{key}"
                # For each folder metric, copy the .datx and .png files to the final output folder, in the corresponding datas_metric and plots_metric folder, depending on the file name contains the metric, e.g. 'He', 'N2', 'O', 'ro', 'Tinf', 'Tz', and also rename the file by replacing the 'DTM20F107Kp' with the key
                for folder_metric in folder_metrics:
                    for filename in os.listdir(unzipped_folder):
                        if folder_metric in filename:
                            if filename.endswith('.datx'):
                                os.system(f"cp {unzipped_folder}/{filename} {final_output_folder}/datas_{folder_metric}/{filename.replace('DTM20F107Kp', key)}")
                            elif filename.endswith('.png'):
                                os.system(f"cp {unzipped_folder}/{filename} {final_output_folder}/plots_{folder_metric}/{filename.replace('DTM20F107Kp', key)}")
        # Save the output_json to the final_output_folder
        with open(f"{final_output_folder}/inputs_runs.json", 'w') as f:
            f.write(str(output_json))
        # Zip the final_output_folder, don't include the parent folder, use the os.system command
        os.system(f"cd {script_dir}/output/{date}/{altitude} && zip -r {final_zip_file} final")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while running the DTM2020 model: {str(e)}")
    
    #check the final zip file
    if os.path.exists(final_zip_file):
        return FileResponse(final_zip_file, media_type='application/octet-stream', filename='final_output.zip')
    else:
        raise HTTPException(status_code=500, detail="An error occurred while creating the final zip file.")