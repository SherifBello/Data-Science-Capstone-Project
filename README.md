# DATA SCIENCE CAPSTONE PROJECT
A Data Science Capstone Project (Exploratory Analysis and Prediction of Cooking Gas Prices in Gombe State) under the Future Tech Fellowship (FutureMap Foundation) program.
## EXPLORATORY ANALYSIS AND PREDICTION OF COOKING GAS PRICES IN GOMBE STATE

**A Data Science Project Report**
  
**Project Type:** Supervised Machine Learning – Regression  
**Primary Algorithm:** Polynomial Linear Regression & Random Forest Regressor
**Programming Language:** Python  
**Libraries:** Pandas, NumPy, Matplotlib, Scikit-learn  
**Target Variable:** LPG Retail Selling Price (NGN/kg)

---

## 1. Overview

This project focuses on developing a machine-learning model to predict the retail selling price of Liquefied Petroleum Gas (LPG) in the Nigerian market especially Gombe State..

The project uses historical LPG sales records combined with historical USD/NGN exchange-rate information to investigate the relationship between economic factors, historical sales patterns and LPG retail prices.

Python is used for data cleaning, data integration, exploratory data analysis, feature engineering, model development and evaluation.

Regression Model was selected as the primary machine-learning algorithm because it provides an interpretable approach to understanding relationships between independent variables and LPG selling prices.

Two modelling approaches are being investigated:

- **Model A:** Random Forest Regressor was used to determine the rate of Sales/predict the rate of sales for some certain period of time.
- **Model B:** Polynomial Linear Regression with features, using the USD/NGN exchange rate, month and day of the week to predict the selling price per kg

The experiment with Model A produced Total Amount MAE of ₦171.5/day, an RMSE of ₦3,315.6/day and an R² score of 0.97.

Model B also produced an MAE of ₦53.4/kg, an RMSE of ₦65.5/kg and an R² score of 0.626

The model that predicts the selling price was deployed using the ONNX runtime web for easier interaction and usage.

## 2. Introduction

Liquefied Petroleum Gas is an important source of energy for households, businesses and industries in Nigeria.

However, LPG retail prices are influenced by several economic and operational factors, including:

- Foreign exchange fluctuations.
- International LPG market prices.
- Domestic supply and demand.
- Depot and wholesale prices.
- Transportation and distribution expenses.
- Importation costs.
- Government policies and market regulations.

Fluctuations in these factors can make LPG pricing difficult to predict.

Machine learning provides an opportunity to analyse historical data, identify patterns and develop predictive models that can support pricing decisions.

This project focuses on developing a machine-learning-based application capable of predicting the retail selling price of LPG per kilogram (₦/kg).

The application uses historical LPG price information and the USD/NGN exchange rate to identify relationships between exchange-rate movements, calendar-related variables, and LPG selling prices.

A Polynomial Linear Regression model was developed using Python and Scikit-learn. After training and validation, the model was converted into the ONNX format and integrated into a browser-based application using JavaScript and ONNX Runtime Web.

The resulting application allows users to provide relevant input variables and obtain an estimated LPG selling price without having to execute Python code.

This project applies data science techniques to historical Gombe LPG sales data to investigate the feasibility of predicting retail LPG prices.

## 3. Problem Statement

LPG retail prices in Nigeria experience fluctuations influenced by economic conditions and supply-chain factors.

Businesses involved in LPG distribution and retail operations need reliable information to understand price movements and support purchasing, inventory and pricing decisions.

Traditional approaches that rely entirely on historical observations and manual market monitoring may not adequately capture relationships between multiple variables.

Therefore, this project investigates how machine learning, particularly Linear Regression and Random Forest Regressor, can be used to predict LPG retail selling prices using historical sales information and exchange-rate data.

## 4. Project Aim and Objectives

### 4.1 Aim

To develop a machine-learning model capable of predicting Gombe LPG retail selling prices using historical sales and Foreign Exchange data.

### 4.2 Objectives

1. Collect and examine historical LPG sales records.
2. Obtain historical USD/NGN exchange-rate data.
3. Clean and preprocess the datasets.
4. Integrate the datasets using their date columns.
5. Aggregate transaction-level records into daily observations.
6. Calculate daily weighted average LPG selling prices.
7. Perform exploratory data analysis.
8. Apply feature engineering to identify useful predictive variables.
9. Develop a regression-based machine-learning model for predicting LPG prices.
10. Evaluate model performance using MAE, RMSE and R².
11. Investigate the relationship between USD/NGN exchange rates and LPG prices.
12. Evaluate the predictive performance of the trained model.
13. Convert the trained model into ONNX format for deployment.
14. Develop a JavaScript-based web application for model inference.
15. Provide an easy-to-use interface for obtaining LPG price predictions.

## 5. Dataset Description

Two primary datasets were used.

### 5.1 LPG Sales Dataset

The original LPG sales dataset were sourced from one of the major gas plant (Salida Gasplant) in Gombe, the files are in daily records of 379 PDF files which were concatenated and contains approximately 20,983 transaction records and 15 columns and converted to csv file format.

It contains historical sales information, including transaction dates, quantities sold, transaction amounts and other operational details.

### 5.2 USD/NGN Exchange-Rate Dataset
The Historical dataset was sourced from (FXConverter.tech) 

The USD/NGN exchange rate represents an important economic variable because exchange-rate movements can influence the cost of imported LPG and associated supply-chain expenses.

The exchange-rate dataset contains 478 daily observations with two columns:

- Date
- USD-NGN exchange rate

And it cover the period from 8th November, 2024 to 19th May, 2026

The exchange-rate variable was introduced to investigate whether fluctuations in the Nigerian naira against the US dollar contribute to changes in domestic LPG retail prices.

### 5.3 Dataset Integration

The datasets were integrated using the Pandas `merge()` function.

The Date column was used as the common key.

A left join was selected to preserve all LPG sales transactions, including dates without a corresponding exchange-rate record.

Missing exchange rates were subsequently handled using 'dropna ()' function.

## 6. Data Preprocessing and Aggregation

The original LPG dataset contained multiple transactions recorded on the same date.

Since the prediction objective focuses on daily LPG selling prices, transaction-level data was aggregated into daily observations.

The weighted average selling price was calculated using:

**Daily LPG Price = Total Daily Revenue / Total Daily Quantity Sold**

Mathematically:

\[
P_t=\frac{\sum Revenue_t}{\sum Quantity_t}
\]

Where:

- \(P_t\) represents the daily weighted average LPG price.
- Revenue represents the total daily sales revenue.
- Quantity represents the total kilograms of LPG sold.

This method accounts for differences in the quantities purchased by customers.

### 6.1 Aggregated Dataset

The resulting daily dataset contains:

| Variable | Description |
|---|---|
| Date | Transaction date |
| Total_Kg | Total kilograms sold |
| Total_Revenue | Total sales revenue in NGN |
| Transaction_Count | Number of transactions |
| USD_NGN | Exchange rate |
| LPG_Price_NGN_KG | Weighted average selling price |

The aggregated dataset contains 295 observations and six columns.

The recorded dates range from 8 November 2024 to 19 May 2026.

These represent dates with recorded sales, not necessarily consecutive calendar days.

## 7. Exploratory Data Analysis

Exploratory Data Analysis (EDA) was used to understand the structure, distribution and behaviour of the dataset.

The main areas of investigation include:

1. LPG retail price movements over time.
2. Changes in USD/NGN exchange rates.
3. Daily sales quantities.
4. Daily revenue patterns.
5. Relationships between exchange rates and LPG prices.

Matplotlib was used to visualise the historical LPG price trend.

The purpose of EDA is to identify patterns and relationships that may help inform the selection of machine-learning features.

## 8. Feature Engineering

Feature engineering involves creating meaningful input variables from existing data to improve machine-learning performance.

Two approaches were considered.

### 8.1 Model A: Random Forest Regressor

This model uses the following features to train the data;

numerical features = Quantity, Measure Unit (Kg), Unit Price (NGN), Month, Day of Week, Day.

Categorical features = Description of Item.

The Target is the Total amount of Sales that occur for each day

### 8.2 Model B: Linear Regression

Model B uses economic and calendar variables without previous LPG prices.

The independent variables are:

| Feature | Description |
|---|---|
| USD_NGN | Exchange rate |
| Month | Month of the year |
| Day_Of_Week | Day of the week |

The dependent variable is:

**Price/Kg**

This approach investigates whether exchange rates and calendar patterns can explain variations in LPG retail prices without relying on previous selling prices.


## 9. Machine Learning Model Development

### 9.1 Model Selection

The project adopted a Linear Regression algorithm enhanced with Polynomial Features of degree 2.

Polynomial Linear Regression was selected to allow the model to capture nonlinear relationships between the input variables and LPG prices.

### 9.2 Polynomial Feature Engineering

Ordinary Linear Regression assumes a linear relationship between input variables and the target.

However, the relationship between exchange rates and LPG prices may not always be strictly linear.

PolynomialFeatures(2) expands the original input variables by generating additional mathematical combinations, including squared terms and interaction terms.

For three input variables, the expanded features include:

USD_NGN

Month

Day_of_Week

USD_NGN²

USD_NGN × Month

USD_NGN × Day_of_Week

Month²

Month × Day_of_Week

Day_of_Week²

The Linear Regression algorithm then learns coefficients for these expanded features.

### 9.3 Model Architecture

The machine-learning pipeline consists of two major components:

1. PolynomialFeatures(degree=2)

Transforms the original three input features into polynomial features.

2. LinearRegression()

Learns the relationship between the transformed features and the LPG selling price.

The complete pipeline is:

PolynomialFeatures(2) → LinearRegression()

### 9.4 Training and Testing

A chronological 80:20 train-test split was adopted.

The first 80% of observations were used for training, while the final 20% were reserved for testing.

This approach preserves the chronological order of the data and avoids randomly mixing historical and future observations.

For Model B, the resulting dataset contained:

- Training observations: 236.
- Testing observations: 59.
- Total usable observations: 295.

## 10. Model Evaluation Metrics

Three evaluation metrics were selected.

### 10.1 Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and predicted LPG prices.

\[
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i|
\]

A lower MAE indicates better predictive accuracy.

### 10.2 Root Mean Squared Error (RMSE)

RMSE measures prediction errors while giving greater weight to larger errors.

\[
RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2}
\]

A lower RMSE indicates better performance.

### 10.3 Coefficient of Determination (R²)

R² measures how much of the variation in the target variable is explained by the model.

\[
R^2=1-\frac{\sum(y_i-\hat y_i)^2}
{\sum(y_i-\bar y)^2}
\]

A value closer to 1 generally indicates stronger explanatory performance.

## 11. Model Results

### 11.1 Model B: Linear Regression

The initial Linear Regression experiment produced the following results:

| Evaluation Metric | Result |
|---|---:|
| MAE | ₦53.4 |
| RMSE | ₦65.5 |
| R² | 0.626 |

Interpretation:

- The MAE indicates an average absolute prediction error of approximately ₦53.4 per kilogram.
- The RMSE indicates the model's root mean squared prediction error was approximately ₦65.5 per kilogram.
- The R² score indicates that the model explained approximately 62.6% of the variation in the test observations.

These results indicate that the model captures some historical price relationships but still has considerable room for improvement.

### 11.2 Model A: Random Forest Regressor

Model A is designed to investigate the predictive contribution of the Numerical features and the Categorical features

| Evaluation Metric | Result |
|---|---:|
| MAE | ₦171.50 |
| RMSE | ₦3,315.59 |
| R² | 0.97 |

Interpretation:

- The MAE indicates an average absolute prediction error of approximately ₦171.50 of Total Sales per Day
- The RMSE indicates the model's root mean squared prediction error was approximately ₦3,315 of Total Sales in a Day.
- The R² score indicates that the model explained approximately 97% of the variation in the test observations.

## 12 MODEL EXPORT AND ONNX CONVERSION

After training and evaluating Model A, the complete Scikit-learn pipeline was converted into the ONNX format.

ONNX (Open Neural Network Exchange) is a model format that supports interoperability between different machine-learning frameworks and deployment environments.

The conversion was performed using:

skl2onnx

onnx

FloatTensorType

The input tensor was defined with three features and a dynamic batch dimension.

Input shape:

[None, 3]

Data type:

float32

The exported model was saved as:

model.onnx

### 12.1 ONNX Model Validation

After exporting the model, the ONNX file was loaded using the ONNX Python package.

The model was checked using:

onnx.checker.check_model(onnx_model)

The validation process returned:

ONNX model is valid!

This confirmed that the exported model passed the ONNX structural validation check.

## 13 APPLICATION DEVELOPMENT AND DEPLOYMENT

The trained ONNX model was integrated into a web application using HTML, CSS, and JavaScript.

The application uses ONNX Runtime Web to execute the trained machine-learning model directly in the browser.

### 13.1 Application Inputs

The application provides three input fields:

USD/NGN Exchange Rate

Month

Day of Week

Users enter or select the required values before initiating a prediction.

### 13.2 Prediction Process

When the user clicks the prediction button, JavaScript collects the three input values.

The values are converted into a Float32Array and passed into an ONNX tensor.

The tensor has the shape:

[1, 3]

The application then executes the ONNX inference session and retrieves the predicted output.

The prediction is displayed on the dashboard in Nigerian Naira per kilogram.

### 13.3 Application Workflow

The application follows this workflow:

The browser loads the web application.

ONNX Runtime Web initializes.

The application loads the trained model.onnx file.

The user supplies the exchange rate, month, and day of the week.

JavaScript converts the inputs into the required tensor format.

ONNX Runtime Web executes the trained model.

The model generates a predicted LPG selling price.

The result is displayed in the application.

### 13.4 SYSTEM ARCHITECTURE

The application consists of two major development stages.

Stage One: Machine Learning Development

Historical LPG and exchange-rate datasets are processed using Python. The selected features are used to train and evaluate the Polynomial Linear Regression model.

The trained pipeline is exported into ONNX format.

Stage Two: Web Application Deployment

The exported ONNX model is loaded into a JavaScript application. Users provide the required input variables, and the model returns a predicted LPG price.

The overall architecture is:

Historical Datasets → Data Preprocessing → Feature Selection → Polynomial Regression → Model Evaluation → ONNX Export → JavaScript Integration → LPG Price Prediction

### 13.5 HOW TO USE THE APPLICATION

The application is designed to be straightforward.

Open the LPG Price Prediction application in a web browser.

Enter the USD/NGN exchange rate.

Select the relevant month.

Select the day of the week.

Click the Predict LPG Price button.

Wait for the model to process the input.

View the estimated LPG selling price displayed in ₦/kg.

The application requires the ONNX model file to be available in the correct location for successful inference.

### 13.6 SYSTEM REQUIREMENTS

Software Requirements

Windows, Linux, or macOS operating system.

Visual Studio Code or another code editor.

Modern web browser.

Python environment for model development.

Required Python machine-learning libraries.

ONNX Runtime Web.

Live Server extension or another local HTTP server.

## 14. Tools and Technologies

The following tools and libraries were used or selected for the project:

| Technology | Application |
|---|---|
| Python | Main programming language |
| Pandas | Data cleaning and manipulation |
| NumPy | Numerical computations |
| Matplotlib | Data visualisation |
| Scikit-learn | Machine-learning development and evaluation |
| Jupyter Notebook | Development environment |
| CSV and Excel | Dataset storage and exchange |
| ONNX | Portable Model Representation |
| HTML | Web Application Structure |
| CSS | User Interface Styling|
| JavaScript | Application Logic and Inference|

## 15. Limitations of the Project

Several limitations should be considered.

1. **Limited observations:** The daily aggregated dataset contains 295 sales observations, which is relatively small for robust forecasting.

2. **Missing economic variables:** International LPG prices, depot prices, transportation costs and supply conditions were not included.

3. **Incomplete calendar coverage:** The dataset contains dates with recorded sales rather than a continuous daily time series.

4. **Exchange-rate representation:** Forward filling assumes the previous available exchange rate remains a reasonable proxy until a new observation becomes available.

5. **Model limitations:** Linear Regression assumes a linear relationship between the selected predictors and the target, which may not fully capture complex LPG market behaviour.

6. **Generalisation:** The model has only been evaluated on a limited historical test period. Performance may change under different market conditions.

## 16. Recommendations

To improve the predictive performance of the project, the following recommendations are proposed:

- Obtain additional historical LPG price records.
- Include Nigerian depot and wholesale prices.
- Incorporate international LPG benchmark prices.
- Investigate transportation and distribution costs.
- Include relevant supply and demand indicators.
- Evaluate additional feature combinations.
- Apply chronological cross-validation.
- Investigate direct multi-horizon forecasting.
- Monitor model performance as new sales data becomes available.

Particular attention should be given to improving the quality and availability of economic variables that influence Nigerian LPG pricing.

## 17. Conclusion

This project demonstrates the application of Python-based data science and supervised machine learning to Gombe LPG retail price prediction.

Historical LPG sales records were integrated with USD/NGN exchange-rate information, cleaned and aggregated into daily observations.

Feature engineering was then applied to develop two Linear Regression modelling approaches: one without lag features and another incorporating historical price information.

A Polynomial Linear Regression model was developed using historical LPG price information and selected economic and calendar-related features.

The trained model was successfully exported into ONNX format and validated. It was subsequently integrated into a browser-based application using HTML, CSS, JavaScript, and ONNX Runtime Web.

The application provides an accessible interface through which users can enter exchange-rate and calendar information to obtain estimated LPG prices per kilogram.

The project demonstrates how machine-learning models developed in Python can be deployed into practical web applications, making predictive analytics more accessible to users in the Nigerian LPG market.

Ultimately, the project aims to develop a practical data-driven approach to understanding and forecasting LPG retail price movements in Gombe.

---


**Primary Research Direction:** Gombe LPG Retail Price Prediction Using Linear Regression

**Prepared as:** Data Science and Machine Learning Project Documentation
