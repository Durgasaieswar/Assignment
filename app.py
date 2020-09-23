import string
import pandas as pd
from models import cos_lav_api1
from models import training
from flask import jsonify
from flask_cors import CORS
from flask import Flask, request


app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
CORS(app)


def clean_string(text):
    """Cleaning text

    Args:
        text (str): Text which needs to be cleaned(Input term)

    Returns:
        text (str): Cleaned text
    """
    # Remove special characters from text
    text = ''.join([char for char in text if char not in string.punctuation])
    return text.lower().strip()


@app.route('/format/match', methods=['POST'])
def string_match():
    """API3 Unsupervised mode

    Returns:
        (JSON): Mapped fields with confidence score
    """
    api_request = request.json
    api_response = {}
    # Collect necessary information from request
    supp_name = api_request['source']['formatName']
    buyer_name = api_request['target']['formatName']
    source_fields = api_request['source']['formatFields']
    target_fields = api_request['target']['formatFields']
    # Find similarity between source & target fields
    mappings = cos_lav_api1.main(source_fields, target_fields)
    # Create response JSON with confidence scores
    api_response['sourceformatName'] = supp_name
    api_response['targetformatName'] = buyer_name
    confidence_score = [i['confidence'] for i in mappings]
    api_response['overallConfidence'] = round(sum(confidence_score)/len(confidence_score), 2)
    api_response['mappings'] = mappings
    return jsonify(api_response)


@app.route('/train/format/learn', methods=['POST'])
def train_model():
    """API2 Learning wrongly mapped fields

    Returns:
        (JSON): Status message with buyer & supplier details
    """
    api_request2 = request.json
    api_response_2 = {}
    # Collect necessary information from request
    source_fields = [i['sourceField'] for i in api_request2['mappings']]
    target_fields = [i['targetField'] for i in api_request2['mappings']]
    df1 = pd.DataFrame()
    # Remove special characters from source & target fields
    source_fields = [clean_string(i) for i in source_fields]
    target_fields = [clean_string(i) for i in target_fields]
    df1['Input'] = source_fields
    df1['Class'] = target_fields
    # print(df1)
    df = pd.read_csv('./training_data.csv')
    # Append newly mapped items to the existing data
    df = df.append(df1, ignore_index=True)
    df.to_csv('./training_data.csv', index=False)
    # Train the model on the new data
    training.train_model(df)
    # Create response JSON
    api_response_2['sourceformatName'] = api_request2['source']['formatName']
    api_response_2['targetformatName'] = api_request2['target']['formatName']
    api_response_2['Message'] = "Learned the mappings"
    return jsonify(api_response_2)


@app.route('/train/format/match', methods=['POST'])
def test_model():
    """API1 Supervised mode

    Returns:
        (JSON): Mapped fields which comes out from model prediction
    """
    api_request3 = request.json
    source_fields = api_request3['source']['formatFields']
    df1 = pd.DataFrame()
    # Remove special characters from source fields
    source_fields = [clean_string(i) for i in source_fields]
    df1['Input'] = source_fields
    # Make predictions for given source fields
    output_df = training.prediction(df1)
    api_response_3, mappings = {}, []
    # Create response JSON with confidence scores & mappings
    api_response_3['sourceformatName'] = api_request3['source']['formatName']
    api_response_3['targetformatName'] = api_request3['target']['formatName']
    api_response_3['overallConfidence'] = round(output_df['confidence'].mean(), 2)
    for ind, row in output_df.iterrows():
        mapped_items = {}
        mapped_items['sourceField'] = row['Input']
        mapped_items['targetField'] = row['predictions']
        mapped_items['confidence'] = row['confidence']
        mappings.append(mapped_items)
    api_response_3['mappings'] = mappings
    return jsonify(api_response_3)


if __name__ == "__main__":
    app.run(threaded=True, debug=False)