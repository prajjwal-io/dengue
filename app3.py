import streamlit as st
import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image  # Added for image processing

@st.cache_resource
def apply_lag(df):
    lag_temp = 16
    lag_rainfall = 4
    lag_dew = 8
    col_name_avg_temp = f'avg_temp_lag_{lag_temp}'
    col_name_total_precipitation = f'rainfall_lag_{lag_rainfall}'
    col_name_avg_dew_temp = f'avg_dew_lag_{lag_dew}'
    df[[col_name_total_precipitation]] = df.groupby('District')[['Total_Precipitation']].shift(lag_rainfall)
    df[[col_name_avg_temp]] = df.groupby('District')[['2m_Temperature']].shift(lag_temp)
    df[[col_name_avg_dew_temp]] = df.groupby('District')[['2m_Dewpoint_Temperature']].shift(lag_dew)
    df = df.sort_values(by=['Record_Year', 'Record_Week'], ascending=[True, True]).reset_index(drop=True)
    df = df.dropna().reset_index(drop=True)
    return df

@st.cache_resource
def encode_and_scale(train_data, test_data, features, target):
    train_data_encoded = pd.get_dummies(train_data, columns=['Record_Week'])
    test_data_encoded = pd.get_dummies(test_data, columns=['Record_Week'])

    X_train, y_train = train_data_encoded[features], train_data_encoded[target]
    X_test, y_test = test_data_encoded[features], test_data_encoded[target]

    # Add constant to the features
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

@st.cache_resource
def train_negative_binomial_model(X_train_scaled, y_train):
    nb_model = sm.GLM(y_train, X_train_scaled, family=sm.families.NegativeBinomial())
    result = nb_model.fit()
    return result

@st.cache_resource
def train_poisson_model(X_train_scaled, y_train):
    poisson_model = sm.GLM(y_train, X_train_scaled, family=sm.families.Poisson())
    result = poisson_model.fit()
    return result

@st.cache_resource
def train_random_forest_model(X_train_scaled, y_train):
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    return rf_model

@st.cache_resource
def train_svr_model(X_train_scaled, y_train):
    # Hyperparameter Tuning for SVR
    param_grid = {
        "C": np.logspace(-2, 3, 6),
        "gamma": np.logspace(-3, 2, 6)
    }
    svr_model = GridSearchCV(SVR(kernel='rbf'), param_grid=param_grid, cv=5)
    svr_model.fit(X_train_scaled, y_train)
    return svr_model

@st.cache_resource
def train_neural_network(X_train_scaled, y_train):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)
    return model

@st.cache_resource
def train_multivariate_regression(X_train_scaled, y_train):
    model = sm.OLS(y_train, X_train_scaled)
    result = model.fit()
    return result

def predict_and_evaluate(result, X_test_scaled, test_data):
    y_pred_nb = result.predict(X_test_scaled)
    test_data['Predicted_Case_Count'] = y_pred_nb

    rmse = np.sqrt(mean_squared_error(test_data['Case_Count'], test_data['Predicted_Case_Count']))
    mae = mean_absolute_error(test_data['Case_Count'], test_data['Predicted_Case_Count'])

    return rmse, mae, test_data

def plot_results(grouped_df):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(grouped_df['Case_Count'], marker='o', linestyle='-', color='mediumblue', label='Actual', linewidth=2)
    ax.plot(grouped_df['Predicted_Case_Count'], marker='o', linestyle='--', color='coral', label='Predicted', linewidth=2)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_xlabel('Record Week', fontsize=12)
    ax.set_ylabel('Case Count', fontsize=12)
    ax.set_title('Actual vs Predicted Case Count for all districts in 2023', fontsize=14)
    ax.legend(fontsize=10)
    legend = ax.legend(fontsize=10)
    legend.get_frame().set_facecolor('lightgray')
    plt.tight_layout()
    return fig

def main():
    # Set the Seaborn style
    sns.set(style="whitegrid", palette="pastel")

    # Load the data
    df = pd.read_csv('data_district.csv')

    # Apply lag to the data
    features2 = apply_lag(df)

    # Separate the dataset into training (2017-2022) and testing (2023) sets
    train_data = features2[features2['Record_Year'] < 2023]
    test_data = features2[features2['Record_Year'] == 2023]

    # Define features and target variable
    all_features = ['Population', 'Geographical_Area', 'Population_Density',
                    'Total_Precipitation', '2m_Dewpoint_Temperature', '2m_Temperature',
                    'rainfall_lag_4', 'avg_temp_lag_16', 'avg_dew_lag_8']
    record_weeks = [f'Record_Week_{i}' for i in range(1, 53)]
    
    # Allow users to choose features
    st.sidebar.header("Select Features")

# Toggle for each feature
    selected_features = {}
    for feature in all_features:
        selected_features[feature] = st.sidebar.checkbox(feature, value=True)

    # Filter out unchecked features
    selected_features = [feature for feature, selected in selected_features.items() if selected]
    features = selected_features + record_weeks
    target = 'Case_Count'


    X_train_scaled, X_test_scaled, y_train, y_test = encode_and_scale(train_data, test_data, features, target)

    # Sidebar option to switch between models
    model_type = st.sidebar.radio("Select Model", ["Negative Binomial", "Poisson", "Random Forest", "SVR", "Neural Network", "Linear Regression"])

    if model_type == "Negative Binomial":
        result = train_negative_binomial_model(X_train_scaled, y_train)
    elif model_type == "Poisson":
        result = train_poisson_model(X_train_scaled, y_train)
    elif model_type == "Random Forest":
        result = train_random_forest_model(X_train_scaled, y_train)
    elif model_type == "SVR":
        result = train_svr_model(X_train_scaled, y_train)
    elif model_type == "Linear Regression":
        result = train_multivariate_regression(X_train_scaled, y_train)

    elif model_type == "Neural Network":
        result = train_neural_network(X_train_scaled, y_train)
        #y_pred = predict_neural_network(nn_model, X_test_scaled)

    rmse, mae, test_data = predict_and_evaluate(result, X_test_scaled, test_data)

    grouped_df = test_data.groupby(['Record_Week']).sum()

    # Set the paths for the logo and background image
    logo_path = "ARTPARKLogo.png"

    @st.cache_data
    def get_img_as_base64(file):
        with open(file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()


    img = get_img_as_base64("Pattern.png")

    # Display background image and set the logo at the top right
    page_bg_img = """
    <style>
    [data-testid="stAppViewContainer"]{
    background-image: url("data:image/png;base64,{img}");
    background-size: cover;
    }
    </style>
    """

    st.markdown(page_bg_img , unsafe_allow_html=True)
    # Display logo
    st.image(logo_path, use_column_width= True, width= 150)

    # Streamlit App
    st.title("Dengue Forecasting App")
    # Display the results plot
    fig = plot_results(grouped_df)
    # Display RMSE and MAE
    st.subheader("Model Evaluation")
    st.text(f"RMSE: {rmse}")
    st.text(f"MAE: {mae}")

    # Show the plot
    st.pyplot(fig)
    

if __name__ == "__main__":
    main()
