import re
import string
import pandas as pd
import numpy as np
from models import cos_lav_api1
from models import training
from flask import jsonify
from flask_cors import CORS
from flask import Flask, request


app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
CORS(app)


# def clean_string(text):
#     """Cleaning text

#     Args:
#         text (str): Text which needs to be cleaned(Input term)

#     Returns:
#         text (str): Cleaned text
#     """
#     # Remove special characters from text
#     new_text = []
#     for char in text:
#         if char in string.punctuation:
#             new_text.append(' ')
#         else:
#             new_text.append(char)
#     text = ''.join(new_text)
#     text = re.sub(r'[0-9]+', ' ', text)
#     return text.lower().strip()


def split_words(source_items):
    """ Split the source item into 2 words

    Args:
        source_items (str): Input terms which requires mapping
    """
    first_items, last_items = [], []
    for source_item in source_items:
        split_items = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]*))|[a-z](?:[a-z]+|[A-Z]*(?=[A-Z]*))', str(source_item).strip())
        if len(split_items) < 2:
            new_source_chars, new_source_item = [], ''
            for letter in source_item:
                if re.search(r'\W', letter):
                    # If special character then add just space
                    new_source_chars.append(' ')
                else:
                    # Just append letter directly
                    new_source_chars.append(letter)
            new_source_item = ''.join(new_source_chars)
            if len(new_source_item.strip().split(' ')) > 1:
                first_items.append(new_source_item.strip().split(' ')[0].lower())
                last_items.append(new_source_item.strip().split(' ')[1].lower())
            else:
                first_items.append(new_source_item.strip().split(' ')[0].lower())
                if '#' in source_item:
                    last_items.append('#')
                else:
                    last_items.append('')
        else:
            first_items.append(split_items[0].lower())
            last_items.append(split_items[1].lower())
    return first_items, last_items


@app.route('/format/match', methods=['POST'])
def string_match():
    """API3 Unsupervised mode

    Returns:
        (JSON): Mapped fields with confidence score
    """
    api_request = request.json
    if api_request:
        api_response = {}
        try:
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
        except Exception as e:
            return jsonify({'status': 'Error occurred while processing'})
    else:
        return jsonify({'status': 'Request payload is not received'})


@app.route('/train/format/learn', methods=['POST'])
def train_model():
    """API2 Learning wrongly mapped fields

    Returns:
        (JSON): Status message with buyer & supplier details
    """
    api_request2 = request.json
    if api_request2:
        api_response_2 = {}
        try:
            # Collect necessary information from request
            source_fields = [i['sourceField'] for i in api_request2['mappings']]
            target_fields = [i['targetField'] for i in api_request2['mappings']]
            df1 = pd.DataFrame()
            df2 = pd.DataFrame()
            # Remove special characters from source & target fields
            source_first, source_last = split_words(source_fields)
            target_first, target_last = split_words(target_fields)
            # source_fields = [clean_string(i) for i in source_fields]
            # target_fields = [clean_string(i) for i in target_fields]
            df1['Input'] = source_first
            df1['Class'] = target_first
            df2['Input'] = source_last
            df2['Class'] = target_last
            # print(df1)
            df_first_item = pd.read_csv('./first_items_data.csv')
            # Append newly mapped items to the existing data
            df_first_item = df_first_item.append(df1, ignore_index=True)
            df_first_item = df_first_item.append(df1, ignore_index=True)
            df_first_item.to_csv('./first_items_data.csv', index=False)
            # Train the model on the new data
            training.train_model(df_first_item, './models/first_items/')
            df_last_item = pd.read_csv('./last_items_data.csv')
            # Append newly mapped items to the existing data
            df_last_item = df_last_item.append(df2, ignore_index=True)
            df_last_item = df_last_item.append(df2, ignore_index=True)
            df_last_item.replace('', np.nan, inplace=True)
            df_last_item.dropna(axis=0, inplace=True)
            df_last_item.to_csv('./last_items_data.csv', index=False)
            training.train_model(df_last_item, './models/last_items/')
            # Create response JSON
            api_response_2['sourceformatName'] = api_request2['source']['formatName']
            api_response_2['targetformatName'] = api_request2['target']['formatName']
            api_response_2['Message'] = "Learned the mappings"
            return jsonify(api_response_2)
        except Exception as e:
            return jsonify({'status': 'Error occurred while processing'})
    else:
        return jsonify({'status': 'Request payload is not received'})


@app.route('/train/format/match', methods=['POST'])
def test_model():
    """API1 Supervised mode

    Returns:
        (JSON): Mapped fields which comes out from model prediction
    """
    api_request3 = request.json
    if api_request3:
        try:
            source_fields = api_request3['source']['formatFields']
            df1 = pd.DataFrame()
            df2 = pd.DataFrame()
            output_df = pd.DataFrame()
            predictions, confidence = [], []
            # Remove special characters from source fields
            # source_fields = [clean_string(i) for i in source_fields]
            predict_first, predict_last = split_words(source_fields)
            df1['Input'] = predict_first
            df2['Input'] = predict_last
            # Make predictions for given source fields
            output_first_df = training.prediction(df1, './models/first_items/')
            output_first_df.to_csv('first.csv', index=False)
            output_last_df = training.prediction(df2, './models/last_items/')
            output_last_df.to_csv('last.csv', index=False)
            for first, last in zip(list(output_first_df['predictions']), list(output_last_df['predictions'])):
                if str(last).strip() in ['nan', '']:
                    predictions.append(str(first))
                    confidence.append((output_first_df.loc[list(output_first_df['predictions']).index(first), 'confidence']))
                else:
                    predictions.append(str(first) + str(last))
                    confidence.append((output_first_df.loc[list(output_first_df['predictions']).index(first), 'confidence'] + output_last_df.loc[list(output_last_df['predictions']).index(last), 'confidence'])/2)
            output_df['Input'] = source_fields
            output_df['predictions'] = predictions
            output_df['confidence'] = confidence
            output_df.to_csv('final.csv', index=False)
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
        except Exception as e:
            return jsonify({'status': 'Error occurred while processing'})
    else:
        return jsonify({'status': 'Request Payload is not received'})

if __name__ == "__main__":
    app.run(threaded=True, debug=False)