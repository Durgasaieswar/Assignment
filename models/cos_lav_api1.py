import re
import string
import Levenshtein
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


def clean_string(text):
    """Cleaning text

    Args:
        text (str): Text which needs to be cleaned(Input term)

    Returns:
        text (str): Cleaned text
    """
    new_text = []
    for char in text:
        if char in string.punctuation:
            new_text.append(' ')
        else:
            new_text.append(char)
    text = ''.join(new_text)
    text = re.sub(r'[0-9]+', ' ', text)
    return text.lower().strip()


def find_cosine_similarity(str1, str2):
    """ Finds cosine similarity between 2 input strings 

    Args:
        str1 (str): sourceField
        str2 (str): targetField

    Returns:
        (float): Similarity score 
    """
    vectorizer = CountVectorizer().fit_transform([str1, str2])
    vectors = vectorizer.toarray()
    vec1 = vectors[0].reshape(1, -1)
    vec2 = vectors[1].reshape(1, -1)
    return (cosine_similarity(vec1, vec2))[0][0]


def find_score(term, string):
    """ Finds the max similarity score between 2 strings 

    Args:
        term (str): sourceField
        string (str): targetField

    Returns:
        similarity_val (float): Maximum similarity value picked from lavenstein score & cosine score
    """
    similarity_val = 0
    term = clean_string(term)
    string = clean_string(string)
    lev_score = Levenshtein.ratio(term, string)
    # Meaning of the word
    cosine_score = find_cosine_similarity(term, string)
    max_score = max([lev_score, cosine_score])
    # print(max_score)
    if max_score is not 0.0:
        similarity_val = max_score
    # returning max among levenshtein and cosine scores  
    return similarity_val


def split_words(source_items):
    """ Split the source item into 2 words

    Args:
        source_item (str): Input term which requires mapping
    """
    first_items, last_items = [], []
    for source_item in source_items:
        split_items = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))|[a-z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', str(source_item).strip())
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
                first_items.append(new_source_item.strip().split(' ')[0])
                last_items.append(new_source_item.strip().split(' ')[1])
            else:
                first_items.append(new_source_item.strip().split(' ')[0])
                last_items.append('#')
        else:
            first_items.append(split_items[0].lower())
            last_items.append(split_items[1].lower())
    return first_items, last_items


def main(input_fields, output_fields):
    """ Trigger the mapping process

    Args:
        input_fields (list): List of sourceFields
        output_fields (list): List of targetFields

    Returns:
        mapp_list (list): List of mapped items with confidence score
    """
    mapp_list = []
    input_1, input_2 = split_words(input_fields)
    print(f'{input_1}, {input_2}')
    for input_term in input_fields:
        confidence, mapp_dict = [], {}
        for output_term in output_fields:
            confidence.append(find_score(input_term, output_term))
        if max(confidence) is not 0:
            map_term_ind = confidence.index(max(confidence))
            mapp_dict['sourceField'] = input_term
            mapp_dict['targetField'] = output_fields[map_term_ind]
            mapp_dict['confidence'] = round(max(confidence)*100, 2)
            mapp_list.append(mapp_dict)
    return mapp_list
    

