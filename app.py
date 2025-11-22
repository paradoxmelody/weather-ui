#import the necessary libraries my lord
#well, since you insist your majesty then I shall do so
#impressive lad aren't you Michael, 
#ooh my Thank you your highness.
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# Page
st.set_page_config(page_title="Weather Temperature Predictor", layout="wide")
st.title("Weather Temperature Predictorrr")
st.write("Enter a date and temperature hints (min/max); click Predict to see the next 3 days. It's that simple!")

# Custom CSS 
st.markdown(
    """
    <style>
    .stApp { background-color: #0D0F1C; color: #E0E6ED; }
    .stBlock { padding-top: 1rem; }
    .big-title { font-size: 30px; font-weight: 700; color: #00C9A7; }
    .metric-box { background-color: #11121A; border-radius: 12px; padding: 16px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_and_prepare_data(path: str = "cleaned_temperature_data.xlsx") -> pd.DataFrame:
    df = pd.read_excel(path)
    # standardize column names from the uploaded file
    if 'DateTime' in df.columns:
        df['DateTime'] = pd.to_datetime(df['DateTime'])
    elif 'Datetime' in df.columns:
        df['DateTime'] = pd.to_datetime(df['Datetime'])
    else:
        raise KeyError("No DateTime column found in data")

    # Handle decimal commas gracefully
    if 'Temperature_Celsius' in df.columns:
        temp_col = 'Temperature_Celsius'
    elif 'Temperature_C' in df.columns:
        temp_col = 'Temperature_C'
    else:
        raise KeyError("No temperature column found in data")

    df[temp_col] = df[temp_col].astype(str).str.replace(',', '.').astype(float)
    df['Temperature_Celsius'] = df[temp_col]
    df['DayOfYear'] = df['DateTime'].dt.dayofyear
    df['Hour'] = df['DateTime'].dt.hour
    df['Date'] = df['DateTime'].dt.date
    return df

try:
    df = load_and_prepare_data()

    @st.cache_resource
    def train_model(df: pd.DataFrame):
        X = df[['DayOfYear', 'Hour']].values
        y = df['Temperature_Celsius'].values
        model = LinearRegression()
        model.fit(X, y)
        return model

    model = train_model(df)

    # Layout inputos
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        input_date = st.date_input("Select date:", value=datetime.now().date())
    with col2:
        min_temp = st.number_input("Minimum Temperature (°C)", value=20.0, step=0.1)
    with col3:
        max_temp = st.number_input("Maximum Temperature (°C)", value=25.0, step=0.1)

    st.markdown("---")

    # Historical visualization: daily averages up to selected date
    hist_df = (
        df[df['DateTime'].dt.date <= input_date]
        .groupby('Date', as_index=False)['Temperature_Celsius']
        .mean()
        .rename(columns={'Temperature_Celsius': 'AvgTemp'})
        .sort_values('Date')
    )

    if len(hist_df) > 0:
        # Show last 60 days of historical averages
        nearest = hist_df.tail(60)   
        chart = (
            alt.Chart(nearest)
            .mark_line(point=True)
            .encode(x='Date:T', y=alt.Y('AvgTemp:Q', title='Avg Temp (°C)'))
            .properties(title='Historical Daily Average Temperature (up to selected date)', height=250)
        )
        st.altair_chart(chart, use_container_width=True)

    if st.button("Gohead and predict the next 3 days"):
        preds = []
        for i in range(1, 4):
            fut = input_date + timedelta(days=i)
            day_of_year = fut.timetuple().tm_yday
            # model predicts representative avg temp at noon
            model_avg = float(model.predict([[day_of_year, 12]])[0])

            # combine user hints with model: weighted average to respect user inputs
            predicted_min = 0.5 * min_temp + 0.5 * (model_avg - 1.0)
            predicted_max = 0.5 * max_temp + 0.5 * (model_avg + 1.0)
            predicted_avg = (predicted_min + predicted_max) / 2.0

            preds.append(
                {
                    'date': fut,
                    'DateLabel': fut.strftime('%a %b %d'),
                    'Min': round(predicted_min, 2),
                    'Max': round(predicted_max, 2),
                    'Avg': round(predicted_avg, 2),
                }
            )

        pred_df = pd.DataFrame(preds)

        st.success('Predictions ready')
        st.markdown('3-Day Temperature Forecast')
        st.dataframe(pred_df[['DateLabel', 'Min', 'Max', 'Avg']].rename(columns={'DateLabel': 'Date'}), use_container_width=True)

        # Plot forecast: area between min and max, and avg line
        forecast_chart = (
            alt.Chart(pred_df)
            .encode(x='date:T')
        )

        area = forecast_chart.mark_area(opacity=0.2, color="#A514E9").encode(
            y='Min:Q', y2='Max:Q'
        )

        avg_line = forecast_chart.mark_line(point=True, color="#751FFF").encode(y='Avg:Q')

        combined = (area + avg_line).properties(height=300, title='Forecast Min/Max and Avg')
        st.altair_chart(combined, use_container_width=True)

        # Nice metric cards
        cols = st.columns(3)
        for idx, row in pred_df.iterrows():
            with cols[idx]:
                st.metric(f"Day {idx+1}", row['DateLabel'])
                c1, c2 = st.columns(2)
                with c1:
                    st.metric('Min (°C)', f"{row['Min']}°C")
                with c2:
                    st.metric('Max (°C)', f"{row['Max']}°C")

except FileNotFoundError:
    st.error("the Data file 'cleaned_temperature_data.xlsx' not found buddy.")
except KeyError as e:
    st.error(f"Problem loading data: {e}")