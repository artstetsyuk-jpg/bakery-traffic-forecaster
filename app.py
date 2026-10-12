import streamlit as st
import pandas as pd
import holidays
from meteostat import daily
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="AI Traffic Forecaster", layout="wide")
st.title("🔮 AI Business Forecaster")
st.write("Upload your historical daily visitor data to instantly train a custom machine learning model.")

# Initialize session state to remember if the AI is trained
if 'ai_trained' not in st.session_state:
    st.session_state.ai_trained = False

# 1. The Drag-and-Drop Uploader
st.sidebar.header("1. Upload Client Data")
uploaded_file = st.sidebar.file_uploader("Drop your CSV file here", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Required format:** A CSV file with `visit_date` and `visitors` columns.")

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    
    if 'visit_date' not in raw_df.columns or 'visitors' not in raw_df.columns:
        st.error("⚠️ Invalid CSV. Please ensure your file has 'visit_date' and 'visitors' columns.")
    else:
        st.success(f"✅ Data loaded successfully! Found {len(raw_df)} days of history.")
        
        if st.sidebar.button("Train AI & Generate Forecast"):
            st.session_state.ai_trained = True
            
        if st.session_state.ai_trained:
            with st.spinner("Processing data & generating forecast..."):
                
                # Standardize the data
                df = raw_df.copy()
                df['visit_date'] = pd.to_datetime(df['visit_date'])
                df = df.sort_values('visit_date')

                # Feature Engineering
                df['day_of_week'] = df['visit_date'].dt.dayofweek
                df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
                df['month'] = df['visit_date'].dt.month
                df['day_of_month'] = df['visit_date'].dt.day
                df['days_in_month'] = df['visit_date'].dt.days_in_month
                # 1. Austrian Payday Culture (Usually strictly end of month, not 1st/15th)
                df['is_payday'] = (df['day_of_month'] >= df['days_in_month'] - 2).astype(int)

                # 2. Vienna Holidays (Province 9 is Wien)
                # 2. French Holidays
                fr_holidays = holidays.FR()
                df['is_holiday'] = df['visit_date'].apply(lambda x: 1 if x in fr_holidays else 0)
                df['is_holiday_tomorrow'] = df['visit_date'].apply(lambda x: 1 if (x + pd.Timedelta(days=1)) in fr_holidays else 0)

                #    ... (Keep your rolling averages and 7-day volatility code exactly the same here) ...

                df['visitors_yesterday'] = df['visitors'].shift(1)
                df['visitors_7d_avg'] = df['visitors'].shift(1).rolling(window=7).mean()
                df['visitors_30d_avg'] = df['visitors'].shift(1).rolling(window=30).mean()
                df['visitors_7d_max'] = df['visitors'].shift(1).rolling(window=7).max()
                df['visitors_30d_max'] = df['visitors'].shift(1).rolling(window=30).max()
                df['visitors_same_day_last_week'] = df['visitors'].shift(7)
                df['days_since_last_visit'] = df['visit_date'].diff().dt.days
                df['post_closure_surge'] = (df['days_since_last_visit'] > 1).astype(int)
                
                df = df.dropna(subset=['visitors_30d_avg'])

                # Fetch Weather
                # 3. Nantes/Guérande Weather Station
                weather_query = daily('07222', df['visit_date'].min(), df['visit_date'].max()).fetch()
                if weather_query is not None and not weather_query.empty:
                    weather_query = weather_query.reset_index()
                    cols = weather_query.columns.tolist()
                    if 'tavg' in cols:
                        df = pd.merge(df, weather_query[['time', 'tavg']], left_on='visit_date', right_on='time', how='left')
                        df['temp'] = df['tavg'].fillna(df['tavg'].mean())
                    else: df['temp'] = 15.0
                    
                    if 'prcp' in cols:
                        df = pd.merge(df, weather_query[['time', 'prcp']], left_on='visit_date', right_on='time', how='left')
                        df['rain'] = df['prcp'].fillna(0)
                    else: df['rain'] = 0
                else:
                    df['rain'], df['temp'] = 0, 15.0

                df['rain_weekend_penalty'] = df['rain'] * df['is_weekend']

                # Train the Model
                features = [
                    'day_of_week', 'is_weekend', 'month', 'day_of_month', 'is_payday',
                    'is_holiday', 'is_holiday_tomorrow', 'temp', 'rain', 'rain_weekend_penalty', 
                    'visitors_yesterday', 'visitors_7d_avg', 'visitors_30d_avg', 
                    'visitors_same_day_last_week', 'post_closure_surge',
                    'visitors_7d_max', 'visitors_30d_max'
                ]
                
                X = df[features]
                y = df['visitors']
                
                # Split without shuffling so our chart shows real chronological dates
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
                model = RandomForestRegressor(n_estimators=200, random_state=42)
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                mae = mean_absolute_error(y_test, predictions)

                # Dashboard Output
                col1, col2, col3 = st.columns(3)
                col1.metric("Average Daily Visitors", f"{int(y.mean())}")
                col2.metric("AI Error Margin (MAE)", f"± {mae:.2f} people")
                col3.metric("Accuracy Score", f"{max(0, 100 - (mae/y.mean() * 100)):.1f}%")

                st.subheader("Actual vs Predicted Traffic (Last 30 Days)")
                
                # FEATURE: Chart UI Polish (Adding real calendar dates to the axis)
                recent_dates = df['visit_date'].iloc[-30:].dt.date
                chart_data = pd.DataFrame({'Actual Visitors': y_test.values[-30:], 'AI Prediction': predictions[-30:]})
                chart_data.index = recent_dates
                st.line_chart(chart_data)

                # FEATURE: Business Translation & 7-Day Forecast
                st.markdown("---")
                st.subheader("📅 7-Day Staffing & Revenue Planner")
                
                # Business translation inputs
                col_b1, col_b2, col_b3, col_b4 = st.columns(4)
                avg_spend = col_b1.number_input("Avg Spend per Guest ($)", value=25.0)
                cust_per_server = col_b2.number_input("Guests per Server", value=15)
                pred_temp = col_b3.slider("Avg Week Temp (°C)", -10.0, 40.0, 20.0)
                pred_rain = col_b4.slider("Avg Week Rain (mm)", 0.0, 50.0, 0.0)
                
                last_row = df.iloc[-1]
                forecast_results = []
                
                # Generate 7 days of predictions
                for i in range(1, 8):
                    tomorrow = last_row['visit_date'] + pd.Timedelta(days=i)
                    
                    future_data = pd.DataFrame([{
                        'day_of_week': tomorrow.dayofweek,
                        'is_weekend': int(tomorrow.dayofweek in [5, 6]),
                        'month': tomorrow.month,
                        'day_of_month': tomorrow.day,
                        'is_payday': int(tomorrow.day in [1, 15] or tomorrow.day >= tomorrow.days_in_month - 1),
                        'is_holiday': int(tomorrow in fr_holidays),
                        'is_holiday_tomorrow': int((tomorrow + pd.Timedelta(days=1)) in fr_holidays),
                        'temp': pred_temp,
                        'rain': pred_rain,
                        'rain_weekend_penalty': pred_rain * int(tomorrow.dayofweek in [5, 6]),
                        'visitors_yesterday': last_row['visitors'] if i == 1 else forecast_results[-1]['Expected Traffic'],
                        'visitors_7d_avg': df['visitors'].tail(7).mean(),
                        'visitors_30d_avg': df['visitors'].tail(30).mean(),
                        'visitors_same_day_last_week': df['visitors'].iloc[-7] if len(df) >= 7 else last_row['visitors'],
                        'post_closure_surge': 0,
                        'visitors_7d_max': df['visitors'].tail(7).max(),
                        'visitors_30d_max': df['visitors'].tail(30).max()
                    }])
                    
                    future_pred = int(model.predict(future_data[features])[0])
                    revenue = f"${future_pred * avg_spend:,.2f}"
                    staff = max(1, (future_pred // cust_per_server) + (1 if future_pred % cust_per_server > 0 else 0))
                    
                    forecast_results.append({
                        'Date': tomorrow.strftime('%A, %b %d'),
                        'Expected Traffic': future_pred,
                        'Est. Revenue': revenue,
                        'Servers Needed': staff
                    })
                
                # Display the planner table
                forecast_df = pd.DataFrame(forecast_results)
                st.dataframe(forecast_df, use_container_width=True, hide_index=True)
                
                # FEATURE: Data Export
                csv_export = forecast_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Schedule (CSV)",
                    data=csv_export,
                    file_name="weekly_forecast_schedule.csv",
                    mime="text/csv"
                )
else:
    st.info("👈 Waiting for data... Please drag and drop a client's CSV file into the sidebar to begin.")