import pickle
import shorttext
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV


def most_scored_class(result):
    """ Get the high probability class name

    Args:
        result (dict) : class names with probablities

    Returns:
        max_class (str): Class name of highest probability
    """
    # print(f'Result: {result}')
    max_score = max(result.values())
    max_index = list(result.values()).index(max_score)
    max_class = list(result.keys())[max_index]
    return max_class


def train_model(df, path):
    """Train the basic model

    Args:
        df (DataFrame): Training data

    Returns:
        (bool): Returns nothing but trained models are stored as pickle files
    """
    trainclassdict = {}
    df_classes = df.groupby(["Class"])
    for key, item in df_classes:
        class_df = df_classes.get_group(key)
        trainclassdict[key] = list(class_df['Input'])
    # Creating topic modeler - word embedding
    num = len(trainclassdict.keys())
    topicmodeler = shorttext.generators.LDAModeler()
    topicmodeler.train(trainclassdict, num)
    topicmodeler.save_compact_model(path + 'topicmodeler.bin')
    # Get cosine similarity
    classifier_cos = shorttext.classifiers.load_gensimtopicvec_cosineClassifier(path + 'topicmodeler.bin')
    cosine_score, kmeans_pred = [], []
    for index, row in df.iterrows():
        result = classifier_cos.score(str(row['Input']))
        max_class = most_scored_class(result)
        cosine_score.append(max_class)
    df['cosine_score'] = cosine_score
    # df.to_csv('./features.csv')
    data = df[['cosine_score']].values
    # OneHotencoding
    enc = preprocessing.OneHotEncoder()
    enc.fit(data)
    # print(enc.categories_)
    data_encoded = enc.transform(data)
    output = open(path + 'onehot_encoder.pkl', 'wb')
    pickle.dump(enc, output)
    output.close()
    labels = list(df['Class'])
    # Splitting training data
    X_train, X_test, y_train, y_test = train_test_split(data_encoded, labels, stratify=labels,
                                                        test_size=0.4)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    # Calibration of trained model to get confidence score for each prediction
    # calibrated_knn = CalibratedClassifierCV(knn, cv="prefit")
    # calibrated_knn.fit(X_train, y_train)
    # pickle.dump(calibrated_knn, open(path + 'calibrated_knn.pkl', 'wb'))
    # Final prediction of source fields
    y_pred = knn.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    pickle.dump(knn, open(path + 'model_one_hot.pkl', 'wb'))
    return True


def prediction(input_terms_df, predict_path):
    """Read the pickle files and make a prediction (Inference code)

    Args:
        input_terms_df (DataFrame): SourceFields

    Returns:
        input_terms_df (DataFrame): Dataframe with source fields & predicted fields
    """
    classifier_cos = shorttext.classifiers.load_gensimtopicvec_cosineClassifier(predict_path + 'topicmodeler.bin')
    # Cosine similarity for new source fields
    for index, row in input_terms_df.iterrows():
        cosine_score = []
        if str(row['Input']).strip() == '#':
            input_terms_df.loc[index, 'predictions'] = 'Reference'
            input_terms_df.loc[index, 'confidence'] = 100.0
        elif str(row['Input']).strip() not in ['nan', '']:
            result = classifier_cos.score(str(row['Input']))
            max_class = most_scored_class(result)
            cosine_score.append(max_class)
            input_terms_df.loc[index, 'cosine_score'] = cosine_score
            # input_terms_df.to_csv('./features.csv')
            # input_data.append(row['Input'])
            test_data = np.array([cosine_score])
            one_encode = pickle.load(open(predict_path + 'onehot_encoder.pkl', 'rb'))
            # OneHotencoding
            test_data = one_encode.transform(test_data)
            knn_predict = pickle.load(open(predict_path + 'model_one_hot.pkl', 'rb'))
            # calibrated_knn = pickle.load(open(predict_path + 'calibrated_knn.pkl', 'rb'))
            # Predictproba to get confidence score for each prediction
            test_predict = knn_predict.predict(test_data)
            probabilities = knn_predict.predict_proba(test_data)
            confid_score = [round(np.amax(probabilities)*100, 2)]
            # for i in probabilities:
            #     print(f'Value: {list(knn_predict.classes_)[np.argmax(i)]}, Confid: {np.amax(i)}')
            #     confid_score.append(np.amax(i))
            input_terms_df.loc[index, 'predictions'] = test_predict
            input_terms_df.loc[index, 'confidence'] = confid_score
    return input_terms_df
