import os
import shutil
import pickle
import pandas as pd
import torch
from chronos import ChronosPipeline
from sklearn.model_selection import train_test_split
from tqdm import tqdm


outputs_dir="/models"
# Load datasets from local files
def load_dataset(file_path):
    return pd.read_excel(file_path, engine='openpyxl')

# Apply column name formatting logic
def format_columns(df):
    df.columns = df.columns.str.replace(r'\(in.*?\)', '', regex=True).str.replace(r'\(S.*?\)', '', regex=True).str.strip()
    return df

# Load and format datasets
activity_df = format_columns(load_dataset('act.xlsx'))
scope_df = format_columns(load_dataset('sc.xlsx'))

# Convert Year and Month to datetime
activity_df['Date'] = pd.to_datetime(activity_df['Year'].astype(str) + '-' + activity_df['Month'].astype(str) + '-01')
scope_df['Date'] = pd.to_datetime(scope_df['Year'].astype(str) + '-' + scope_df['Month'].astype(str) + '-01')

# Train Chronos model
def train_chronos_model(train_data, column, prediction_length=12, num_samples=20):
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    context = torch.tensor(train_data[column].values, dtype=torch.float32).unsqueeze(0)
    forecast = pipeline.predict(
        context=context,
        prediction_length=prediction_length,
        num_samples=num_samples,
    )
    return pipeline, forecast

def train_and_store_models(data_df,data_type):
    models = {}
    for (compId, location_name), location_group in tqdm(data_df.groupby(['compId', 'Location']), desc=f"Training {data_type.capitalize()} Models"):
        if compId not in models:
            models[compId] = {}
        models[compId][location_name] = {}
        for column in tqdm(location_group.columns, desc=f"Processing {location_name}", leave=False):
            if column in ['Date', 'compId', 'Location', 'Year', 'Month']:
                continue
            if pd.api.types.is_numeric_dtype(location_group[column]):
                try:
                    # Check if we have enough data points
                    if len(location_group) > 1:
                        train_data, _ = train_test_split(location_group, test_size=0.2, shuffle=False)
                    else:
                        # If we only have one data point, use it as is
                        train_data = location_group
                    
                    model, forecast = train_chronos_model(train_data, column)
                    compId_dir = f"{outputs_dir}/{compId}"
                    
                    location_dir = f"{compId_dir}/{location_name}"
                    if not os.path.exists(location_dir):
                        os.makedirs(location_dir, exist_ok=True)
                    model_path = f"{location_dir}/{column}_model.pt"
                    torch.save(model.model.state_dict(), model_path)
                    models[compId][location_name][column] = {
                        'model_path': model_path,
                        'forecast': forecast.cpu().numpy().tolist()
                    }
                except Exception as e:
                    print(f"Error training model for {location_name}, {column}: {e}")
    return models


def clear_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def pickle_and_save_models(models, data_type):
    for compId in models:
        compId_dir = f"{outputs_dir}/{compId}"
        os.makedirs(compId_dir, exist_ok=True)
        with open(f'{compId_dir}/{data_type}_models.pkl', 'wb') as f:
            pickle.dump(models[compId], f)
            
 # Train models
clear_directory(outputs_dir)
activity_models = train_and_store_models(activity_df, "activity")
scope_models = train_and_store_models(scope_df, "scope")
 
        # Save models locally
pickle_and_save_models(activity_models, "activity")
pickle_and_save_models(scope_models, "scope")