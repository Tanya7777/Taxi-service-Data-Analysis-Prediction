# Taxi-service-Data-Analysis-Prediction

This project focuses on performing end-to-end exploratory data analysis (EDA), feature engineering, and machine learning modeling on taxi service trip records to uncover insights and forecast future demand. The goal is to help taxi businesses make data-driven decisions by understanding ride patterns and predicting key metrics like trip duration, fare amount, and pickup demand.

---

## Table of Contents

- [Overview](#overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Tools & Technologies](#tools--technologies)
- [Data Science Workflow](#data-science-workflow)
- [Key Findings](#key-findings)
- [Models Used](#models-used)
- [Installation & Usage](#installation--usage)
- [Future Work](#future-work)


---

## Overview

Taxi companies generate large volumes of data through GPS-enabled trip logs. This project leverages such data to extract business insights and develop predictive models that can assist in:

- Identifying peak hours, high-demand zones  
- Predicting trip duration based on route and time  
- Forecasting fare amounts using geospatial and time data  
- Understanding seasonality and customer behavior  

---

## Objectives

- Perform in-depth EDA on historical taxi trip data  
- Visualize trip patterns using geospatial and time-based plots  
- Build regression models to predict fare and trip duration  
- Evaluate model performance and compare techniques  
- Translate findings into actionable insights for stakeholders  

---

## Dataset

The dataset used for this project is based on the official [NYC Taxi Trip Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), consisting of millions of records with the following fields:

- Pickup & Drop-off datetime  
- Pickup & Drop-off latitude and longitude  
- Trip distance and duration  
- Passenger count  
- Fare amount, payment type  
- Vendor ID  

> Note: The dataset was cleaned and filtered for modeling.

---

## Tools & Technologies

- **Programming**: Python  
- **Data Analysis**: Pandas, NumPy  
- **Visualization**: Matplotlib, Seaborn, Plotly, Folium  
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM  
- **Geospatial Analysis**: Geopandas, Folium  
- **Notebook Environment**: Jupyter Notebook  

---

## Data Science Workflow

1. **Data Cleaning**  
   - Removed nulls, duplicates, and invalid GPS coordinates  
   - Handled outliers in trip duration and fare  

2. **Exploratory Data Analysis (EDA)**  
   - Time series analysis of trip counts  
   - Fare vs. distance relationships  
   - Demand heatmaps by time and location  

3. **Feature Engineering**  
   - Extracted date-time features: hour, weekday, month  
   - Calculated Haversine distance between pickup and drop-off  
   - Added traffic proxy using timestamp clustering  

4. **Model Development**  
   - Trained and tuned multiple regression models  
   - Compared performance using RMSE, MAE, and R²  

5. **Evaluation & Interpretation**  
   - Analyzed feature importance  
   - Visualized prediction errors and outliers  

---

## Key Findings

- Peak ride demand occurs during weekday rush hours (8–10 AM, 5–7 PM)  
- Trips shorter than 3 km tend to have higher per-km fare due to base charges  
- Downtown and airport zones see highest ride density  
- Distance and time of day are strong predictors of fare  

---

## Models Used

| Model         | Purpose                   | Performance Metric      |
|---------------|----------------------------|--------------------------|
| Linear Regression | Baseline regression     | MAE / RMSE               |
| Random Forest    | Non-linear predictions  | Lower RMSE than baseline |
| XGBoost          | Optimized regression    | Best R² performance      |
| LightGBM         | Fast training, scalable | Competitive accuracy     |

---

## Installation & Usage

1. **Clone the repository**
```bash
git clone https://github.com/Tanya7777/Taxi-service-Data-Analysis-Prediction.git
cd Taxi-service-Data-Analysis-Prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Launch the notebook**
```bash
jupyter notebook taxi_analysis.ipynb
```

---

## Future Work

- Deploy fare prediction model as a REST API  
- Add real-time map-based ride prediction dashboard  
- Incorporate weather and traffic data into features  
- Improve geospatial clustering for route optimization  





