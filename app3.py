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


# Load your background image
background_image_path = "IISc.png"
# Load your logo image
logo_path = "ARTPARKLogo.png"

# Convert image to base64
with open(background_image_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

page_bg_img = f"""
<style>
[data-testid="stApp"]{{
    
    background-color: #f5f5f5;
    background-size: cover;
    background-repeat: no-repeat;
}}
[data-testid="stHeader"]{{
background: rgba(0,0,0,0);

}}
[data-testid="stSidebar"] {{

    background : rgb(58, 152, 185);
    background-attachment: fixed;
    background-size: cover;
    background-repeat: no-repeat;
    background-position: top-right;
    
    }}

</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)


@st.cache_resource
def apply_lag(df):
    lag_temp = 16
    lag_rainfall = 4
    lag_dew = 8
    col_name_avg_temp = 'Avg_Temperature'
    col_name_total_precipitation = 'Rainfall'
    col_name_avg_dew_temp = 'Avg_Dewpoint_Temperature'
    df[[col_name_total_precipitation]] = df.groupby('District')[['Total_Precipitation']].shift(lag_rainfall)
    df[[col_name_avg_temp]] = df.groupby('District')[['Temperature']].shift(lag_temp)
    df[[col_name_avg_dew_temp]] = df.groupby('District')[['Dewpoint_Temperature']].shift(lag_dew)
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

def plot_results(grouped_df, test_data):
    fig, ax = plt.subplots(figsize=(12, 6))
    rmse = np.sqrt(mean_squared_error(test_data['Case_Count'], test_data['Predicted_Case_Count']))
    mae = mean_absolute_error(test_data['Case_Count'], test_data['Predicted_Case_Count'])
    ax.plot(grouped_df['Case_Count'], marker='o', linestyle='-', color='mediumblue', label='Actual', linewidth=2)
    ax.plot(grouped_df['Predicted_Case_Count'], marker='o', linestyle='--', color='coral', label='Predicted', linewidth=2)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_xlabel('Record Week', fontsize=12)
    ax.set_ylabel('Case Count', fontsize=12)
    ax.set_title('Actual vs Predicted Case Count for all districts in 2023', fontsize=14)
    ax.legend(fontsize=10)
    legend = ax.legend(fontsize=10)
    legend.get_frame().set_facecolor('lightgray')
    # Annotate RMSE and MAE values
    bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
    ax.annotate(f"RMSE: {rmse:.2f}", xy=(1.0, 0.70), xycoords='axes fraction', ha="right", va="center",
                bbox=bbox_props, fontsize=10)
    ax.annotate(f"MAE: {mae:.2f}", xy=(1.0, 0.62), xycoords='axes fraction', ha="right", va="center",
                bbox=bbox_props, fontsize=10)
    plt.tight_layout()
    return fig


def plot_district_results(test_data, district_name, predicted_column):
    district_data = test_data[test_data['District'] == district_name]
    #add rmse and mae values on the top of graph
    rmse = np.sqrt(mean_squared_error(district_data['Case_Count'], district_data[predicted_column]))
    mae = mean_absolute_error(district_data['Case_Count'], district_data[predicted_column])

    #annotate rmse and mae values on the top of graph
    
    if district_data.empty:
        print(f"District data for {district_name} is empty.")
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(district_data['Record_Week'], district_data['Case_Count'], marker= 'o', linestyle = '-', color = 'mediumblue', label='Actual', linewidth=2)
    ax.plot(district_data['Record_Week'], district_data[predicted_column], marker='o', linestyle='--', color='coral', label='Predicted', linewidth=2)
    ax.set_xlabel('Record Week', fontsize=12)
    ax.set_ylabel('Case Count', fontsize=12)
    ax.set_title(f'Actual vs Predicted Case Count for {district_name} in 2023)', fontsize=14)
    ax.legend(fontsize=10)
    legend = ax.legend(fontsize=10)
    legend.get_frame().set_facecolor('lightgray')
    # Annotate RMSE and MAE values
    bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
    ax.annotate(f"RMSE: {rmse:.2f}", xy=(1.0, 0.70), xycoords='axes fraction', ha="right", va="center",
                bbox=bbox_props, fontsize=10)
    ax.annotate(f"MAE: {mae:.2f}", xy=(1.0, 0.62), xycoords='axes fraction', ha="right", va="center",
                bbox=bbox_props, fontsize=10)
    plt.tight_layout()

    return fig



def main():
    # Set the Seaborn style
    sns.set(style="whitegrid", palette="pastel")

    # Load the data
    df = pd.read_csv('data_district.csv')

    #change the name of columns
    df = df.rename(columns={'2m_Dewpoint_Temperature': 'Dewpoint_Temperature', '2m_Temperature': 'Temperature', 'rainfall_lag_4': 'Rainfall', 'avg_temp_lag_16': 'Avg_Temperature', 'avg_dew_lag_8': 'Avg_Dewpoint_Temperature'})

    # Apply lag to the data
    features2 = apply_lag(df)

    # Separate the dataset into training (2017-2022) and testing (2023) sets
    train_data = features2[features2['Record_Year'] < 2023]
    test_data = features2[features2['Record_Year'] == 2023]

    # Define features and target variable
    all_features = ['Population', 'Geographical_Area', 'Population_Density',
                    'Total_Precipitation', 'Dewpoint_Temperature', 'Temperature',
                    'Rainfall', 'Avg_Temperature', 'Avg_Dewpoint_Temperature']

    # all_features = ['Population', 'Geographical_Area', 'Population_Density',
    #                 'Total_Precipitation', '2m_Dewpoint_Temperature', '2m_Temperature',
    #                 'rainfall_lag_4', 'avg_temp_lag_16', 'avg_dew_lag_8']
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
    st.sidebar.header("Select Model")

    model_type = st.sidebar.radio("List : ", ["Negative Binomial", "Poisson", "Random Forest", "SVR", "Neural Network", "Linear Regression"] , label_visibility="visible")

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
    print(test_data)

    grouped_df = test_data.groupby(['Record_Week']).sum()
   
    # Display logo
    st.image(logo_path, use_column_width= True, width= 150)

    # Streamlit App
    st.title("Dengue Forecasting App" , anchor= "center")
    #change the color of the title
    st.markdown(""" <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style> """, unsafe_allow_html=True)
    #st.markdown('<style>body{background-color: #E5E7E9;}</style>',unsafe_allow_html=True)
    st.markdown('<style>h1{color: #1F618D;}</style>',unsafe_allow_html=True)
        # Allow users to choose the district
    selected_district = st.sidebar.selectbox("Select District", test_data['District'].unique(), index=0)
    # Display the results plot
    fig = plot_results(grouped_df, test_data)
    # Display the district-specific plot
    if selected_district:
        fig2 = plot_district_results(test_data, selected_district, 'Predicted_Case_Count')
        st.pyplot(fig2)
    # Show the plot
    st.pyplot(fig)
    #st.pyplot(fig2)
    # # Display RMSE and MAE at the bottom right
    # rmse_col, mae_col = st.columns([3, 1])
    # with rmse_col:
    #     st.write("")
    # with mae_col:
    #     st.text(f"RMSE: {rmse}")
    #     st.text(f"MAE: {mae}")


if __name__ == "__main__":
    main()
