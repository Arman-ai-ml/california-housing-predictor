from flask import Flask, request, render_template_string
import pickle
import numpy as np
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

# Load ML model
with open(BASE_DIR / 'california_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load scaler
with open(BASE_DIR / 'scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


PAGE = """
<h2>California House Price Predictor</h2>

<form method="POST">
    <input name="MedInc" placeholder="Median Income" type="number" step="any" required><br><br>
    <input name="HouseAge" placeholder="House Age" type="number" step="any" required><br><br>
    <input name="AveRooms" placeholder="Average Rooms" type="number" step="any" required><br><br>
    <input name="AveBedrms" placeholder="Average Bedrooms" type="number" step="any" required><br><br>
    <input name="Population" placeholder="Population" type="number" step="any" required><br><br>
    <input name="AveOccup" placeholder="Average Occupancy" type="number" step="any" required><br><br>
    <input name="Latitude" placeholder="Latitude" type="number" step="any" required><br><br>
    <input name="Longitude" placeholder="Longitude" type="number" step="any" required><br><br>

    <button type="submit">Predict Price</button>
</form>

{% if prediction is not none %}
<h3>Predicted House Value: {{ prediction }}</h3>
{% endif %}

{% if error %}
<p>{{ error }}</p>
{% endif %}
"""


@app.route('/', methods=['GET', 'POST'])
def home():

    prediction = None
    error = None

    if request.method == 'POST':
        try:
            features = [
                float(request.form['MedInc']),
                float(request.form['HouseAge']),
                float(request.form['AveRooms']),
                float(request.form['AveBedrms']),
                float(request.form['Population']),
                float(request.form['AveOccup']),
                float(request.form['Latitude']),
                float(request.form['Longitude'])
            ]

            input_data = np.array(features).reshape(1, -1)
            scaled_data = scaler.transform(input_data)

            result = model.predict(scaled_data)[0]
            prediction = round(float(result), 3)

        except Exception as e:
            error = str(e)

    return render_template_string(
        PAGE, prediction=prediction, error=error
    )


if __name__ == '__main__':
    app.run(debug=True)


from flask import Flask, request, render_template_string
import pickle
import numpy as np
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / 'california_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open(BASE_DIR / 'scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

DEFAULT_VALUES = {
    'MedInc': '8.3252',
    'HouseAge': '41',
    'AveRooms': '6.9841',
    'AveBedrms': '1.0238',
    'Population': '322',
    'AveOccup': '2.5556',
    'Latitude': '37.88',
    'Longitude': '-122.23'
}

PAGE = """
<h2>California House Price Predictor</h2>

<form method="POST">

<label>Median Income</label><br>
<input name="MedInc" value="{{ values['MedInc'] }}" type="number" step="any" required><br><br>

<label>House Age</label><br>
<input name="HouseAge" value="{{ values['HouseAge'] }}" type="number" step="any" required><br><br>

<label>Average Rooms</label><br>
<input name="AveRooms" value="{{ values['AveRooms'] }}" type="number" step="any" required><br><br>

<label>Average Bedrooms</label><br>
<input name="AveBedrms" value="{{ values['AveBedrms'] }}" type="number" step="any" required><br><br>

<label>Population</label><br>
<input name="Population" value="{{ values['Population'] }}" type="number" step="any" required><br><br>

<label>Average Occupancy</label><br>
<input name="AveOccup" value="{{ values['AveOccup'] }}" type="number" step="any" required><br><br>

<label>Latitude</label><br>
<input name="Latitude" value="{{ values['Latitude'] }}" type="number" step="any" required><br><br>

<label>Longitude</label><br>
<input name="Longitude" value="{{ values['Longitude'] }}" type="number" step="any" required><br><br>

<button type="submit">Predict Price</button>

</form>

{% if prediction is not none %}
<h3>Predicted House Value: {{ prediction }}</h3>
{% endif %}

{% if error %}
<p>{{ error }}</p>
{% endif %}
"""

@app.route('/', methods=['GET', 'POST'])
def home():

    values = DEFAULT_VALUES.copy()
    prediction = None
    error = None

    if request.method == 'POST':

        values = {
            key: request.form.get(key, DEFAULT_VALUES[key])
            for key in DEFAULT_VALUES
        }

        try:
            features = np.array([
                [float(values[key]) for key in DEFAULT_VALUES]
            ])

            scaled_data = scaler.transform(features)
            result = model.predict(scaled_data)[0]

            prediction = round(float(result), 3)

        except Exception as e:
            error = str(e)

    return render_template_string(
        PAGE,
        values=values,
        prediction=prediction,
        error=error
    )


if __name__ == '__main__':
    app.run(debug=True)

    