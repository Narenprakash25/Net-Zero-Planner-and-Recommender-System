import os
import pickle
import pandas as pd
import torch
from chronos import ChronosPipeline
import numpy as np
from io import BytesIO
from openai import AzureOpenAI
import json
from datetime import datetime
from json.decoder import JSONDecodeError
import asyncio
import streamlit as st
import io
import plotly.express as px
import plotly.graph_objects as go

client = AzureOpenAI(
    azure_endpoint="https://carboncompeteazureopenai.openai.azure.com/",
    api_key="7bc8f94429fe48da9fbe33487b224974",
    api_version="2024-02-01"
)

# Load datasets from local files
def load_dataset(file_path):
    return pd.read_excel(file_path, engine='openpyxl')

# Load and format datasets
activity_df = load_dataset('act.xlsx')
scope_df = load_dataset('sc.xlsx')

# Convert Year and Month to datetime
activity_df['Date'] = pd.to_datetime(activity_df['Year'].astype(str) + '-' + activity_df['Month'].astype(str) + '-01')
scope_df['Date'] = pd.to_datetime(scope_df['Year'].astype(str) + '-' + scope_df['Month'].astype(str) + '-01')

# Load unique compIds and locations
unique_compIds = activity_df['compId'].unique()
unique_locations = activity_df['Location'].unique()

# Load models
def load_models(compId):
    compId_dir = f"outputs/{compId}"
    with open(f'{compId_dir}/scope_models.pkl', 'rb') as f:
        scope_models = pickle.load(f)
    with open(f'{compId_dir}/activity_models.pkl', 'rb') as f:
        activity_models = pickle.load(f)
    return scope_models, activity_models

async def load_chronos_model(model_info):
    try:
        if isinstance(model_info, dict) and 'model_path' in model_info:
            model_path = model_info['model_path']
        else:
            model_path = model_info

        with open(model_path, 'rb') as f:
            model_data = BytesIO(f.read())

        pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small",
            device_map="cpu",
            torch_dtype=torch.float32,
        )
        pipeline.model.load_state_dict(torch.load(model_data, weights_only=True))  # Use weights_only=True
        return pipeline
    except Exception as e:
        return None

def make_forecast(model, historical_data, periods=12, freq='M'):
    context = torch.tensor(historical_data['y'].values).unsqueeze(0)
    forecast = model.predict(
        context=context,
        prediction_length=periods,
        num_samples=20,
        limit_prediction_length=False  
    )
    future_dates = pd.date_range(start=historical_data['Date'].max() + pd.Timedelta(days=1), periods=periods, freq='M')
    
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecast': forecast.mean(axis=1).numpy().flatten(),
        'Lower Bound': forecast.quantile(0.05, axis=1).numpy().flatten(),
        'Upper Bound': forecast.quantile(0.95, axis=1).numpy().flatten()
    })
    
    forecast_df['Forecast'] = forecast_df['Forecast'].apply(lambda x: max(x, 0))
    forecast_df['Lower Bound'] = forecast_df['Lower Bound'].apply(lambda x: max(x, 0))
    forecast_df['Upper Bound'] = forecast_df['Upper Bound'].apply(lambda x: max(x, 0))
    
    return forecast_df

def create_json_response(historical_data, forecast_data):
    min_year = historical_data['Date'].dt.year.min()
    max_year = forecast_data['Date'].dt.year.max()
    
    response = {
        "historical": {},
        "forecast": {}
    }
    
    months = ['January', 'February', 'March', 'April', 'May', 'June', 
              'July', 'August', 'September', 'October', 'November', 'December']
    
    # Historical data
    for year in range(min_year, max_year + 1):
        year_data = historical_data[historical_data['Date'].dt.year == year]
        response["historical"][str(year)] = {month: "" for month in months}
        
        if not year_data.empty:
            for _, row in year_data.iterrows():
                month = row['Date'].strftime('%B')
                response["historical"][str(year)][month] = str(row['y'])
                
    # Forecast data
    for year in range(forecast_data['Date'].dt.year.min(), forecast_data['Date'].dt.year.max() + 1):
        year_data = forecast_data[forecast_data['Date'].dt.year == year]
        response["forecast"][str(year)] = {
            month: {
                "value": "",
                "upperbound": "",
                "lowerbound": ""
            } for month in months
        }
        
        for _, row in year_data.iterrows():
            month = row['Date'].strftime('%B')
            response["forecast"][str(year)][month] = {
                "value": str(row['Forecast']),
                "upperbound": str(row['Upper Bound']),
                "lowerbound": str(row['Lower Bound'])
            }
    
    return response

def calculate_yearly_totals(activity_df, scope_df, compId, location):
    # Filter data for the specific compId and location
    activity_filtered = activity_df[(activity_df['compId'] == compId) & (activity_df['Location'] == location)]
    scope_filtered = scope_df[(scope_df['compId'] == compId) & (scope_df['Location'] == location)]
    
    # Exclude non-numeric columns for summation
    numeric_activity_columns = activity_filtered.select_dtypes(include=[np.number]).columns
    numeric_scope_columns = scope_filtered.select_dtypes(include=[np.number]).columns
    
    # Calculate yearly totals for activity data
    activity_yearly = activity_filtered.groupby('Year')[numeric_activity_columns].sum()
    
    # Calculate yearly totals for scope data
    scope_yearly = scope_filtered.groupby('Year')[numeric_scope_columns].sum()
    
    # Combine activity and scope yearly totals
    yearly_totals = pd.merge(activity_yearly, scope_yearly, left_index=True, right_index=True, suffixes=('_activity', '_scope'))
    
    # Reset index to make 'Year' a column again
    yearly_totals = yearly_totals.reset_index()
    
    return yearly_totals

async def process_data_in_batches(data_df, models, selected_location, months_to_forecast, columns_to_process, batch_size=4):
    results = {}

    async def load_model(column, model_info):
        return column, await load_chronos_model(model_info)

    columns_batches = [columns_to_process[i:i + batch_size] for i in range(0, len(columns_to_process), batch_size)]
    
    for batch in columns_batches:
        model_load_tasks = [load_model(column, models[selected_location][column]) for column in batch]
        loaded_models = dict(await asyncio.gather(*model_load_tasks))

        async def process_column(column):
            model = loaded_models.get(column)
            if model is not None:
                historical_data = data_df[data_df['Location'] == selected_location][['Date', column]].rename(columns={column: 'y'})
                loop = asyncio.get_event_loop()
                forecast_data = await loop.run_in_executor(
                    None, 
                    make_forecast,
                    model, 
                    historical_data, 
                    months_to_forecast
                )
                result = create_json_response(historical_data, forecast_data)
                return column, result
            return column, None

        # Process all columns in the batch concurrently
        batch_results = dict(await asyncio.gather(*[process_column(column) for column in batch]))
        results.update(batch_results)

        # Clear memory after processing each batch
        del loaded_models
        del model_load_tasks
        del batch_results

    return results

async def generate_scenario_recommendations(scenario, filtered_activity_df_csv, filtered_scope_df_csv, yearly_totals_csv, location):
    # First, load the yearly totals into a DataFrame
    
    instructions = f"""
    You are the Sustainability Head of a company with over 60 years of experience in scenario-based recommendations. Analyze the historical data and yearly totals for {location}'s activity usage and scope emissions. Focus on the following tasks:

    1. Identification of Significant Contributors:
    - Identify the primary activities and emissions that significantly contribute to the total emissions for {location}.
    - Ensure that these identified contributors have a consistent significant impact across the years.
    - If an initially identified contributor shows minimal impact in recent years, exclude it and select the next highest consistent contributor.
    - Rank the contributors by their impact on total emissions and select the top contributors regardless of their order in the list.

    2. Trend Analysis:
    - Examine the historical data for the identified significant contributors for both scope and activity data.
    - Provide clear, concise descriptions of the trends for each significant contributor in complete sentences.

    3. Recommendations and Targets:
    - Based on the identified significant contributors, provide tailored recommendations to meet the {scenario['Target']} goal.
    - Use the latest year's total (from the yearly totals data) as the baseline for each significant contributor.
    - Generate yearly targets from the latest year to {scenario['Target'].split()[-1]} for these significant activities and emissions, ensuring a logical progression towards the final goal.

    Ensure all analyses, recommendations, and targets are specific to {location} and aligned with the provided historical data and yearly totals.

    IMPORTANT:
    1. Focus ONLY on the significant contributors identified for {location}.
    2. Provide trends and analysis in complete sentences, not single words.
    3. In the targets section, provide numerical values that are STRICTLY based on the historical data trends and yearly totals.
    4. Ensure there are a maximum of 4 contributors and not less than 3 contributors for Scope 1, Scope 2, and Scope 3 emissions combined.
    5. When referring to activities or emissions sources, use the EXACT column names as they appear in the provided CSV data.
    6. Carefully examine the data for ALL scopes (Scope 1, Scope 2, and Scope 3). Do not assume any scope is zero without thorough verification.

    Scope Definitions:
    1. Scope 1: Emissions that a company owns or controls directly, such as from burning fuel in vehicles, chemical production, or emissions from smokestacks.
    2. Scope 2: Emissions that a company causes indirectly, such as from the generation of electricity, steam, heating, and cooling that the company purchases and uses.
    3. Scope 3: Emissions that occur in the upstream and downstream activities of a company, such as from the consumption or use of a company's goods or services.

    Note:
    - DIESEL contributes to Scope 1 emissions (this is combustion of diesel in equipment not logistics)
    - EB_POWER contributes to Scope 2 emissions.
    - CARTON_BOX, FV_DUMP, METAL_SCRAP, PLASTIC_CRATE, DRY_WASTE, FRESH_WATER, DIESEL_FIRST, DIESEL_LAST, (here the DIESEL_FIRST and DIESEL_LAST are fuel used in logistics) CNG_FIRST, CNG_LAST contribute to Scope 3 emissions.
    - If DIESEL is included in Scope 1 emissions should not stop you from including DIESEL_FIRST and DIESEL_LAST as primary contributors to Scope 3 emissions if they have significant contributions than other activities which contribute to Scope 3 emissions. 
    - All other activities like Treated Water, RE Power, Energy Saved, RE Capacity Installed, Water Saved, EV_first and EV_last have no contribution to emissions.

    
    Please provide your response as a JSON object with the following structure:

    {{
        "Location": "{location}",
        "Scenarios": [
            {{
                "info": {{
                    "scenarioType": "{scenario['Scenario Type']}",
                    "industryType": "{scenario['Industry Type']}",
                    "targetYear": "{scenario['Target'].split()[-1]}",
                    "recommendationType": "{scenario['Recommendation Type']}",
                    "scenarioNumber": "1"
                }},
                "analysis": {{
                    "scopeEmissions": [
                        {{
                            "scope": "Scope 1",
                            "trend": "",
                            "contributors": []
                        }},
                        {{
                            "scope": "Scope 2",
                            "trend": "",
                            "contributors": []
                        }},
                        {{
                            "scope": "Scope 3",
                            "trend": "",
                            "contributors": []
                        }}
                    ],
                    "activityData": [
                        {{
                            "activity": "",
                            "trend": "",
                            "unit": ""
                        }}
                    ],
                    "primaryContributors": []
                }},
                "recommendations": [
                    {{
                        "category": "",
                        "actions": [
                            {{
                                "action": "",
                                "impact": {{
                                    "currentUsage": "",
                                    "targetReduction": "",
                                    "expectedEmissionReduction": ""
                                }},
                                "calculations": ""
                            }}
                        ]
                    }}
                ],
                "targets": {{
                    "activities": {{
                        "activityName": {{
                            "currentYear": "",
                            "yearlyTargets": {{}}
                        }}
                    }},
                    "emissions": {{
                        "scopeName": {{
                            "currentYear": "",
                            "yearlyTargets": {{}}
                        }}
                    }}
                }},
                "conclusion": {{
                    "feasibility": "",
                    "alternativeTargets": ""
                }}
            }}
        ]
    }}

    Please ensure the following in your response:

    1. Focus on the verified Highest 4 major contributing activities for {location} in the activityData and targets.
    2. Provide yearly targets from the latest year to {scenario['Target'].split()[-1]} for both activities and emissions, ensuring a logical progression towards the final goal.
    3. Align all recommendations with the {scenario['Target'].split()[-1]} goal, tailoring them specifically to {location}'s context and needs.
    4. Format the JSON response to be directly parseable, adhering to proper JSON syntax and structure.
    5. Include detailed calculations demonstrating the projected impact of each recommendation.
    6. Specify units consistently throughout the analysis (e.g., diesel and EB power in GJ, water in kL, waste in tonnes, Diesel, CNG, First and Last travel in km, emissions in tCO2e).
    7. Present all trends, analyses, and recommendations in clear, complete sentences that provide comprehensive context.
    8. Ensure targets for activities and emissions are based on historical data and yearly totals, demonstrating a realistic year-over-year reduction path to meet the {scenario['Target'].split()[-1]} goal.
    9. Targets should be provided as numerical values without units, and should be on the same scale as the historical data and yearly totals.

    CRITICAL:
    - Ensure that all yearly targets in your response are directly based on and aligned with the historical data and yearly totals provided, accounting for the full range of variations.
    - Ensure that the 'currentYear' value for each scope and activity EXACTLY matches the latest year's total from the yearly totals data provided. Double-check all calculations before including them in your response.
    - For activities or emissions that have zero or very low values (less than 20% of the maximum value) in the yearly totals, set the 'currentYear' value to 0 or the actual small value, and exclude these from your recommendations and targets.
    - The yearly progression of targets MUST reflect both the historical trends and the necessary reductions to meet the final goal.
    - Do NOT generate targets that are significantly out of scale with the yearly totals.
    - When referring to activities or emissions sources, you MUST use the EXACT column names as they appear in the provided CSV data. This includes using underscores where present (e.g., 'Diesel_used', 'EB_power'). Do not abbreviate or modify these names in any way.
    - After generating your response, validate all 'currentYear' values against the provided yearly totals data. If any discrepancies are found, recalculate and update your response accordingly.

    Your response should reflect a deep understanding of {location}'s specific challenges and opportunities in achieving the designated sustainability goals.

    Use the provided historical data and yearly totals to inform your analysis and recommendations, focusing only on data relevant to {location}.
    """

    prompt = f"""
    {instructions}

    Location: {location}
    Industry Type: {scenario['Industry Type']}
    Scenario Type: {scenario['Scenario Type']}
    Target: {scenario['Target']}
    Recommendation Type: {scenario['Recommendation Type']}
    Historical Scope Data: {filtered_scope_df_csv}
    Historical Activity Data: {filtered_activity_df_csv}
    Yearly Totals: {yearly_totals_csv}

    Please provide your analysis, recommendations, and targets in the specified JSON format, focusing only on activities and emissions relevant to {location}.
    """


    response = await asyncio.to_thread(
        client.chat.completions.create,
        model="gpt-4o",
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def clean_and_parse_json(json_string):
    # Remove any leading or trailing non-JSON content
    start = json_string.find('{')
    end = json_string.rfind('}') + 1
    if start != -1 and end != -1:
        json_string = json_string[start:end]
    
    try:
        return json.loads(json_string)
    except JSONDecodeError:
        # If JSON parsing fails, return a structured error message
        return {
            "error": "Failed to parse JSON",
            "raw_output": json_string
        }

def plot_emission(scope_name, historical_data, forecast_data):
    # Prepare historical data
    hist_records = []
    for year, months in historical_data.get(scope_name, {}).items():
        for month, value in months.items():
            if value != "":
                date_str = f"{month} {year}"
                date = pd.to_datetime(date_str, format="%B %Y")
                hist_records.append({'Date': date, 'Value': float(value)})

    hist_df = pd.DataFrame(hist_records)

    # Prepare forecast data
    forecast_records = []
    for year, months in forecast_data.get(scope_name, {}).items():
        for month, data in months.items():
            if data['value'] != "":
                date_str = f"{month} {year}"
                date = pd.to_datetime(date_str, format="%B %Y")
                forecast_records.append({
                    'Date': date,
                    'Forecast': float(data['value']),
                    'Upper Bound': float(data['upperbound']),
                    'Lower Bound': float(data['lowerbound'])
                })

    forecast_df = pd.DataFrame(forecast_records)

    # Combine data
    combined_df = pd.merge(hist_df, forecast_df, on='Date', how='outer')
    combined_df = combined_df.sort_values('Date')

    # Plot
    fig = go.Figure()

    if not hist_df.empty:
        fig.add_trace(go.Scatter(
            x=hist_df['Date'], y=hist_df['Value'], mode='lines+markers', name='Historical'
        ))

    if not forecast_df.empty:
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'], y=forecast_df['Forecast'], mode='lines+markers', name='Forecast'
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'], y=forecast_df['Upper Bound'], mode='lines', name='Upper Bound', line=dict(dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'], y=forecast_df['Lower Bound'], mode='lines', name='Lower Bound', line=dict(dash='dash')
        ))

    fig.update_layout(title=f'Emission Forecast for {scope_name}', xaxis_title='Date', yaxis_title='Value')
    st.plotly_chart(fig)

def plot_activity(activity_name, historical_data, forecast_data):
    # Prepare historical data
    hist_records = []
    for year, months in historical_data.get(activity_name, {}).items():
        for month, value in months.items():
            if value != "":
                date_str = f"{month} {year}"
                date = pd.to_datetime(date_str, format="%B %Y")
                hist_records.append({'Date': date, 'Value': float(value)})

    hist_df = pd.DataFrame(hist_records)

    # Prepare forecast data
    forecast_records = []
    for year, months in forecast_data.get(activity_name, {}).items():
        for month, data in months.items():
            if data['value'] != "":
                date_str = f"{month} {year}"
                date = pd.to_datetime(date_str, format="%B %Y")
                forecast_records.append({
                    'Date': date,
                    'Forecast': float(data['value']),
                    'Upper Bound': float(data['upperbound']),
                    'Lower Bound': float(data['lowerbound'])
                })

    forecast_df = pd.DataFrame(forecast_records)

    # Combine data
    combined_df = pd.merge(hist_df, forecast_df, on='Date', how='outer')
    combined_df = combined_df.sort_values('Date')

    # Plot
    fig = go.Figure()

    if not hist_df.empty:
        fig.add_trace(go.Scatter(
            x=hist_df['Date'], y=hist_df['Value'], mode='lines+markers', name='Historical'
        ))

    if not forecast_df.empty:
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'], y=forecast_df['Forecast'], mode='lines+markers', name='Forecast'
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'], y=forecast_df['Upper Bound'], mode='lines', name='Upper Bound', line=dict(dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'], y=forecast_df['Lower Bound'], mode='lines', name='Lower Bound', line=dict(dash='dash')
        ))

    fig.update_layout(title=f'Activity Forecast for {activity_name}', xaxis_title='Date', yaxis_title='Value')
    st.plotly_chart(fig)

async def main():
    st.title('Emissions Forecasting and Scenario Recommendations')

    # Input form
    st.sidebar.header('Input Parameters')
    compId = st.sidebar.selectbox('Company ID', options=unique_compIds)
    location = st.sidebar.selectbox('Location', options=unique_locations)
    scenarioType = st.sidebar.text_input('Scenario Type')
    industryType = st.sidebar.text_input('Industry Type')
    targetYear = st.sidebar.number_input('Target Year', min_value=datetime.now().year)
    recommendationType = st.sidebar.text_input('Recommendation Type')
    percentage = st.sidebar.number_input('Reduction Percentage (optional)', min_value=0, max_value=100, step=1)

    if st.sidebar.button('Generate Forecast and Recommendations'):
        with st.spinner('Loading models and data...'):
            # Load models and data
            scope_models, activity_models = load_models(compId)
            st.success('Models and data loaded successfully')

        scenario = {
            'Location': location,
            'Scenario Type': scenarioType,
            'Industry Type': industryType,
            'Target': f"Net Zero by {targetYear}" if scenarioType == 'Net Zero' else f"{percentage}% by {targetYear}",
            'Recommendation Type': recommendationType,
        }

        # Filter data for the specific compId and location
        filtered_activity_df = activity_df[(activity_df['Location'] == location) & (activity_df['compId'] == compId)]
        filtered_scope_df = scope_df[(scope_df['Location'] == location) & (scope_df['compId'] == compId)]

        filtered_activity_df_csv = filtered_activity_df.to_csv(index=False)
        filtered_scope_df_csv = filtered_scope_df.to_csv(index=False)

        # Calculate yearly totals
        yearly_totals_df = calculate_yearly_totals(filtered_activity_df, filtered_scope_df, compId, location)
        yearly_totals_csv = yearly_totals_df.to_csv(index=False)

        last_recorded_date = max(scope_df['Date'].max(), activity_df['Date'].max())
        months_to_forecast = (pd.Timestamp(year=targetYear, month=12, day=31) - last_recorded_date).days // 30

        # Generate scenario recommendations
        scenario_json = await generate_scenario_recommendations(
            scenario, filtered_activity_df_csv, filtered_scope_df_csv, yearly_totals_csv, location
        )
        scenario_data = clean_and_parse_json(scenario_json)
        primary_contributors = scenario_data["Scenarios"][0]["analysis"]["primaryContributors"]

        # Use primary contributors for activity columns
        scope_columns = ['Scope_1_Emissions', 'Scope_2_Emissions', 'Scope_3_Emissions']
        activity_columns = [contributor for contributor in primary_contributors if contributor in filtered_activity_df.columns]

        # Process data in batches
        initial_forecast = await process_data_in_batches(
            scope_df, scope_models, location, months_to_forecast, scope_columns, batch_size=4
        )
        activity_forecast = await process_data_in_batches(
            activity_df, activity_models, location, months_to_forecast, activity_columns, batch_size=4
        )

        final_results = {
            "location": location,
            "scenarios": scenario_data["Scenarios"],
            "data": {
                "emissions": {
                    "historical": {scope: data["historical"] for scope, data in initial_forecast.items()},
                    "forecast": {scope: data["forecast"] for scope, data in initial_forecast.items()}
                },
                "activities": {
                    "historical": {activity: data["historical"] for activity, data in activity_forecast.items()},
                    "forecast": {activity: data["forecast"] for activity, data in activity_forecast.items()}
                }
            }
        }

        # Display scenario information
        st.header('Scenario Information')
        scenario_info = final_results['scenarios'][0]['info']
        st.write(f"**Location:** {final_results['location']}")
        st.write(f"**Scenario Type:** {scenario_info['scenarioType']}")
        st.write(f"**Industry Type:** {scenario_info['industryType']}")
        st.write(f"**Target Year:** {scenario_info['targetYear']}")
        st.write(f"**Recommendation Type:** {scenario_info['recommendationType']}")

        # Display analysis
        st.header('Analysis')
        analysis = final_results['scenarios'][0]['analysis']
        # For scope emissions
        st.subheader('Scope Emissions')
        for scope in analysis['scopeEmissions']:
            st.write(f"**{scope['scope']}**")
            st.write(scope['trend'])
            if 'contributors' in scope and scope['contributors']:
                st.write('Contributors:')
                for contributor in scope['contributors']:
                    st.write(f"- {contributor}")

        # For activity data
        st.subheader('Activity Data')
        if 'activityData' in analysis and analysis['activityData']:
            for activity in analysis['activityData']:
                st.write(f"**{activity['activity']}**")
                st.write(activity['trend'])
                if 'unit' in activity:
                    st.write(f"Unit: {activity['unit']}")

        # Display recommendations
        st.header('Recommendations')
        recommendations = final_results['scenarios'][0]['recommendations']
        for rec in recommendations:
            st.subheader(rec['category'])
            for action in rec['actions']:
                st.write(f"**Action:** {action['action']}")
                impact = action.get('impact', {})
                if impact:
                    st.write("Impact:")
                    st.write(f"- Current Usage: {impact.get('currentUsage', '')}")
                    st.write(f"- Target Reduction: {impact.get('targetReduction', '')}")
                    st.write(f"- Expected Emission Reduction: {impact.get('expectedEmissionReduction', '')}")
                st.write(f"**Calculations:** {action.get('calculations', '')}")

        # Display conclusions
        st.header('Conclusion')
        conclusion = final_results['scenarios'][0]['conclusion']
        st.write(f"**Feasibility:** {conclusion.get('feasibility', '')}")
        st.write(f"**Alternative Targets:** {conclusion.get('alternativeTargets', '')}")

        # Plot emissions
        st.header('Emission Forecasts')
        emissions_historical = final_results['data']['emissions']['historical']
        emissions_forecast = final_results['data']['emissions']['forecast']
        for scope_name in emissions_historical.keys():
            st.subheader(f'{scope_name}')
            plot_emission(scope_name, emissions_historical, emissions_forecast)

        # Plot activities
        st.header('Activity Forecasts')
        activities_historical = final_results['data']['activities']['historical']
        activities_forecast = final_results['data']['activities']['forecast']
        for activity_name in activities_historical.keys():
            st.subheader(f'{activity_name}')
            plot_activity(activity_name, activities_historical, activities_forecast)

if __name__ == "__main__":
    asyncio.run(main())
