import re
import pandas as pd
from configs.config import path
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity

def read_data(paths):
    datas = pd.read_csv(paths)
    return datas.head(5)

def data_preprocess(df):
    df_yt = df.drop_duplicates(subset=['title'])
    df_yt = df_yt[['title', 'description']]
    df_yt['cleaned_title'] = df_yt['title'].apply(lambda x: x.lower())
    df_yt['cleaned_title'] = df_yt['cleaned_title'].apply(lambda x: re.sub('[^A-Za-z0-9]+', ' ', x))
    return df_yt


def load_bert_models():
    preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1", trainable=True)
    return preprocessor, encoder

def get_bert_embeddings(text, preprocessor, encoder):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    encoder_inputs = preprocessor(text_input)
    outputs = encoder(encoder_inputs)
    embedding_model = tf.keras.Model(text_input, outputs['pooled_output'])
    sentences = tf.constant([text])
    return embedding_model(sentences)

def preprocess_text():
    text = input()
    text = text.lower()
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    return text

def calculate_similarity(processed_data, query_encoding):
    processed_data['similarity_score'] = processed_data['encodings'].apply(
        lambda x: cosine_similarity(x.numpy().reshape(1, -1), query_encoding.numpy().reshape(1, -1))[0][0]
    )
    return processed_data.sort_values(by=['similarity_score'], ascending=False)


def main():
    data = read_data(path)
    processed_data = data_preprocess(data)
    print("Processed Data:\n", processed_data)

    preprocessor, encoder = load_bert_models()
    processed_data['encodings'] = processed_data['cleaned_title'].apply(
        lambda x: get_bert_embeddings(x, preprocessor, encoder)
    )
    query = preprocess_text()
    query_encoding = get_bert_embeddings(query, preprocessor, encoder)
    df_results = calculate_similarity(processed_data, query_encoding)
    print("\nResults sorted by similarity score:\n", df_results)

if __name__ == "__main__":
    main()
