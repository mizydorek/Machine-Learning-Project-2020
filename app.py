# Numerical arrays.
import numpy as np

# Data manipulation and analysis.
import pandas as pd 

# Import Flask.
import flask as fl 
from flask import request, jsonfiy make_response


# Import LinearRegression model from sklearn linear model.
from sklearn.linear_model import LinearRegression
# Using train_test_split() from the scikit-learn library, 
# makes it easy to split dataset into training and testing data.
from sklearn.model_selection import train_test_split

# create a web app. 
app = fl.Flask(__name__)

# Add routes.
@app.route('/')
def hello():
    return app.send_static_file('index.html')

@app.route('/api/normal')
def normal():
    return { 'value': np.random.normal() }

@app.route('/api/linear', methods=["POST"])
def model():

    req = request.get_json()

    print(req)

    response = make_response(jsonify({ 'value': np.random.normal() }), 200)

    #x = np.random.choice(26)

    #{ 'value': linear(x) }
    return response

# Load dataset function.
def load():
    try:
        # Load dataset.
        df = pd.read_csv("https://raw.githubusercontent.com/ianmcloughlin/2020A-machstat-project/master/dataset/powerproduction.csv")
        # Drop columns for power equals 0 above 1m/s wind speed.
        df = df.drop(df[(df.speed > 1) & (df.power == 0)].index).reset_index(drop=True)
        # Split dataframe into speed and power columns using pandas.
        speed = df['speed']
        power = df['power']

        # Convert speed and power columns to numpy arrays.
        x = df.iloc[:, :1].values 
        y = df.iloc[:, 1].values

        return x, y, speed, power
    except:
        print("Failed to load dataset.")

# Preprocessing of dataset
def preprocess():
    try:
        # load data.
        x, y, speed, power = load()
        # Split the dataset into 75% train data and 25% test data.
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
        return x_train, x_test, y_train, y_test
    except:
        print("Failed to preprocess data.")

# Build linear model.
def linear(x):
    try:
        # load preprocess data.
        x_train, x_test, y_train, y_test = preprocess()
        # Create a new linear regression model.
        model = LinearRegression()
        # Fit the data to the model.
        model.fit(x_train, y_train);
        # Coefficient & intercept.
        coeff = [model.coef_[0], model.intercept_]
        # Function that count the linear regression.
        return x * coeff[0] + coeff[1]
    except:
        print("Failed to bulid and train the model.")