from flask import request, jsonify, Flask, render_template
from functions import load_data

app = Flask(__name__)

df = load_data('customers_data_light.csv')
data = df.to_dict()
data = data['target']

@app.route('/', methods=['GET'])
def home():
    return '''<h1>Distant Model Results</h1>
<p>An API for distant reading of our Home Credit Risk Scoring Model.</p>'''

# A route to return all of the available scoring entries in our dataset.
@app.route('/api/v1/resources/df/all', methods=['GET'])
def api_alldf():
    return data

@app.route('/api/v1/resources/targets', methods=['GET'])

def api_id():
    # Check if an ID was provided as part of the URL.
    # If ID is provided, assign it to a variable.
    # If no ID is provided, display an error in the browser.
    if 'id' in request.args:
        id = int(request.args['id'])
    else:
        return "Error: No id field provided. Please specify an id."

    # Create an empty list for our results
    results = []

    # Loop through the data and match results that fit the requested ID.
    # IDs are unique, but other fields might return many results
    for target in data:
        if target == id:
            results.append(data[id])

    # Use the jsonify function from Flask to convert our list of
    # Python dictionaries to the JSON format.
    #return render_template('index.html', variable=id, target = data[id])
    return jsonify(results)

@app.route('/api/v1/resources/results', methods=['GET'])

def api_id2():
    # Check if an ID was provided as part of the URL.
    # If ID is provided, assign it to a variable.
    # If no ID is provided, display an error in the browser.
    if 'id' in request.args:
        id = int(request.args['id'])
    else:
        return "Error: No id field provided. Please specify an id."

    # Create an empty list for our results
    results = []

    # Loop through the data and match results that fit the requested ID.
    # IDs are unique, but other fields might return many results
    for target in data:
        if target == id:
            results.append(data[id])

    # Use the jsonify function from Flask to convert our list of
    # Python dictionaries to the JSON format.
    return render_template('index.html', variable=id, target = data[id])
    #return jsonify(results)

if __name__ == "__main__":
    #app.run(debug=True)
    app.run(host="151.80.119.47", port=5000, threaded=True, debug=True)
