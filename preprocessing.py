import numpy as np
import pandas as pd
import time 
from datetime import datetime
import re
import numpy as np
from os import remove
import argparse
import matplotlib.pyplot as plt

# Text libraries 
import regex as re
import html
import fasttext
import tldextract
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Import dictionaries 
from emot.emo_unicode import EMOTICONS_EMO
from utils import SLANGS, CONTRACTIONS, STOPWORDS

# Import classifier
from utils import MODELS, MODEL_NAMES, PARAMETERS_TREES, PARAMETERS_MPL
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier




# Resampling 
from imblearn.over_sampling import SMOTE


#MODELS = [ RandomForestClassifier(random_state=42), LogisticRegression(random_state=42), NearestNeighbors(), DecisionTreeClassifier(random_state=42)]
#MODEL_NAMES= ["SVC", "Random Forest", "Logistic Regression", "KNN", "DecisionTreeClassifier"]


def resampling(X_train, y_train):

    # Resample the minority class. You can change the strategy to 'auto' if you are not sure.
    sm = SMOTE(sampling_strategy='auto', random_state=7)

    # Fit the model to generate the data.
    oversampled_trainX, oversampled_trainY = sm.fit_resample(X_train, y_train)
    #oversampled_train = pd.concat([pd.DataFrame(oversampled_trainY), pd.DataFrame(oversampled_trainX)], axis=1)

   # nr_label_0, nr_label_1, nr_label_2 = df.loc[df["airline_sentiment"] == 0].shape[0],  df.loc[df["airline_sentiment"] == 1].shape[0],  df.loc[df["airline_sentiment"] == 2].shape[0] 
    #nr_label_0_oversampled, nr_label_1_oversampled, nr_label_2_oversampled = df.loc[df["airline_sentiment"] == 0].shape[0],  df.loc[df["airline_sentiment"] == 1].shape[0],  df.loc[df["airline_sentiment"] == 2].shape[0] 

    # print(f"Shape starting df: 0:{nr_label_0}, 1:{nr_label_1}, 2: {nr_label_2}")
    # print(f"Shape starting df: 0:{nr_label_0_oversampled}, 1:{nr_label_1_oversampled}, 2: {nr_label_2_oversampled}")
    
    return oversampled_trainX, oversampled_trainY 



def tuning_classifiers(clf, parameters_grid, X_train, y_train, X_val, y_val):
    print("Starting tuning classifiers")

    f1_scores = list()
    parameters_grid = ParameterGrid(parameters_grid)

    for config in parameters_grid:
        clf.set_params(**config)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

        f1 = f1_score(y_val, y_pred, average = "macro")
        f1_scores.append(f1)
    
    print(f"Max f1 score: {max(f1_scores)}\n\n")
    best_config = list(parameters_grid)[np.argmax(np.array(f1_scores))]
    print(f"Best configuration: {best_config}")

    return best_config
        

    #
    # cv = GridSearchCV(clf, parameters_grid, cv=k_fold, scoring="f1_macro", n_jobs=-1).fit(X_train,y_train)
 
    #print(cv.best_estimator_)
    #print(cv.best_score_)



def test_models(X_train : pd.DataFrame, y_train : pd.Series, X_val : pd.DataFrame, y_val : pd.Series ) -> pd.DataFrame:
    # Test 4 models and print results 
   
    f1_scores_val= []
    f1_scores_train = []

    for clf, name in zip(MODELS, MODEL_NAMES):

        #print(f'Start training model: {name}\n\n')

        start_time = time.time()
        clf.fit(X_train, y_train)

        y_pred_val =clf.predict(X_val)
        f1_val = f1_score(y_val, y_pred_val, average='macro')
        
        y_pred_train =clf.predict(X_train)
        f1_train = f1_score(y_train, y_pred_train, average='macro')

        finish_time = time.time()

        print(f'Finishing training model: {name}, trained in {finish_time-start_time:.2f}')

        f1_scores_train.append(f1_train)
        f1_scores_val.append(f1_val)

        print(f'Score of {name} model performed: {f1_val:.2f}\n\n\n')

        col1 = pd.Series(MODEL_NAMES)
        col2 = pd.Series(f1_scores_val)
        col3 = pd.Series(f1_scores_train)

        if name == "Random Forest":
            fig, ax = plt.subplots(figsize=(15,12))
            feature_importances = pd.DataFrame(clf.feature_importances_, index = X_train.columns,  columns=['importance']).sort_values('importance', ascending=False)[:5]
            feature_importances.plot(kind = "barh", ylabel ="importances", color = "SeaGreen", grid = True,  ecolor ="black", ax = ax)
        
            fig.savefig("FastText.png")

            print(feature_importances)

    result = pd.concat([col1, col2, col3], axis = 'columns')
    result.columns = ['Model Name', 'F1 Score Val', "F1 Score Train"]
    
    print(result)

    return result



def add_word_embeddings(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame) -> tuple([pd.DataFrame, pd.DataFrame]):
    """Add the word embeddings scores to the development and evaluation set, based on the vocabolary of the training set.
    Parameters
    ----------
    X_valid_train : pd.DataFrame
        Devolopment set dataframe
    X_valid_test : pd.DataFrame
        Evaluation set dataframe
    Returns
    -------
    tuple([pd.DataFrame, pd.DataFrame, pd.Series])
        A tuple with X_train, X_test and y_train
    """

    print('Starting word embeddings')
    
    text_train = np.array("__label__") + y_train.astype("str").values + np.array(" ") + X_train["text"].values
    
    np.savetxt("train.txt", text_train, fmt="%s")   
    
    model = fasttext.train_supervised("train.txt", lr=0.1, epoch=50, wordNgrams=3, dim=50, loss='ova')
    model.save_model("model.bin")

    remove("train.txt")
    
    
    # Create features 
    scores_train = []
    scores_val = []


    for text in X_train["text"]:
        prediction = model.predict(text, k=3)
        scores_train.append(map(lambda x : x[1], sorted(zip(prediction[0],prediction[1]), key=lambda x : x[0])))
        

    for text in X_val["text"]:
        prediction = model.predict(text, k=3)
        scores_val.append(map(lambda x : x[1], sorted(zip(prediction[0],prediction[1]), key=lambda x : x[0])))
      


    new_features_names= ["embedding_negativity", "embedding_positivity", "embedding_neutr"]

    scores_train = pd.DataFrame(scores_train, columns=new_features_names)
    scores_val = pd.DataFrame(scores_val, columns=new_features_names)

    
    X_train = pd.concat([X_train, scores_train], axis=1)
    X_val = pd.concat([X_val, scores_val], axis=1)
    
    return X_train, X_val



def text_tf_idf(X_train: pd.DataFrame, X_test: pd.DataFrame, min_df=0.01) -> tuple([pd.DataFrame, pd.DataFrame]):
    """Applies the tf-df to the text feature of the train and the evaluation set
    Parameters
    ----------
    X_train : pd.DataFrame
        Devolopment set dataframe
    X_test : pd.DataFrame
        Evaluation set dataframe
    min_df : float
        Minimum support for the tf-df. If is between (0, 1) is percentual, if it is >1 its absolute value is considered
    Returns
    -------
    tuple([pd.DataFrame, pd.DataFrame])
        A tuple with X_train and X_test with the tf-df applied
    """
    
    vectorizer = TfidfVectorizer(strip_accents="ascii", stop_words = STOPWORDS, use_idf=False, min_df=min_df)
    vectorizer.fit(X_train["text"])
    train_tfdf = pd.DataFrame(vectorizer.transform(X_train["text"]).toarray(), columns=vectorizer.get_feature_names())
    test_tfdf = pd.DataFrame(vectorizer.transform(X_test["text"]).toarray(), columns=vectorizer.get_feature_names())

    # Concat features obtained by TF-DF with the original DataFrame
    X_train = pd.concat([X_train.reset_index(drop=True),train_tfdf.reset_index(drop=True)], axis=1)
    X_test = pd.concat([X_test.reset_index(drop=True),test_tfdf.reset_index(drop=True)], axis = 1) 
    return X_train, X_test



def expand_contraction_form(text:str) -> str:
    # Replace contractions forms with a string that has the equivalent meaning by using CONTRACTIONS dic in utils
    for word in text.split(sep= " "):
        if word in CONTRACTIONS.keys():
            text = re.sub(word , CONTRACTIONS[word], text)
        
    return text



def convert_slangs(text:str) -> str:
    # Replace slangs in the text by using the SLANGS dict
    for word in text.split(sep= " "):
        if word in SLANGS.keys():
            text = re.sub(word , SLANGS[word], text)
    return text



def convert_emoticons(text:str) -> str:
    # Replace emoticons in the text such as ':)' by using the EMOTICONS_EMO dict
    for emot in EMOTICONS_EMO:
        text = re.sub(re.escape(emot), EMOTICONS_EMO[emot], text)
    return text



def tweet_tokenize(line:str) -> str:
    # Tokenize the text
    tokenize_text = TweetTokenizer().tokenize(line) 
    return tokenize_text



def clean_text(tweets: pd.DataFrame) -> pd.Series:
    """Clean the text format and remove irrelevant informations
    Parameters
    ----------
    tweets : pd.DataFrame
        Dataframe of tweets

    Returns
    -------
    pd.DataFrame
        Same DataFrame with text cleaned 
    """
    
    # Convert text in lower case
    tweets["text"] = tweets["text"].str.lower()

    # Remove the tag "@" in the tweets
    tweets["text"] = list(map((lambda x : re.sub("@\w+", "", x)), tweets['text']))

    #  Expand contracted form
    tweets["text"] = list(map(lambda x: expand_contraction_form(x),tweets['text'] ))

    # Convert HTML entities into characters
    tweets["text"] = list(map(lambda x: html.unescape(x), tweets['text'])) 

    # Extract the domain from the URLs
    tweets["text"] = tweets['text'].str.replace(pat="((https?:\/\/)?([w]+\.)?\S+)", repl=lambda x: tldextract.extract(x.group(1)).domain, regex=True)

    # Remove repeated characters
    tweets["text"] = list(map(lambda x: re.sub("(\w)\\1{2,}", "\\1\\1", x), tweets['text']))

    # Remove numbers 
    tweets["text"] = list(map(lambda elem: re.sub(r"\d+", "", elem), tweets['text']))

    # Replace emoticons into text
    tweets["text"] = list(map(lambda x: convert_emoticons(x), tweets["text"]))

    # Replace slangs
    tweets["text"] = list(map(lambda x: convert_slangs(x), tweets["text"]))
 
    # Tokenize text with Tweeter tokenizer
    tweets["text"] = tweets["text"].apply(lambda x: " ".join(tweet_tokenize(x)))
    
    return tweets["text"]



def check_US(state: str, dict_US_State) -> int:
    country = False

    if state in dict_US_State:
        country = True
  
    return country



def pre_processing():

    df = pd.read_csv("data/Tweets.csv") # read from CSV file

    
    # Drop columns
    drop_columns = ["tweet_id", "airline_sentiment_confidence", "negativereason", "negativereason_confidence",\
                 "airline_sentiment_gold", "name", "tweet_coord", "user_timezone", "negativereason_gold", "tweet_location", "retweet_count", "tweet_location"]

    df.drop(columns = drop_columns, inplace=True)
    
    
    # 1. Convert datetime in timestamp 
    pattern = " -0800"
    df["tweet_created"] = df["tweet_created"].apply(lambda x : re.sub(pattern, '', x ))\
                                             .apply(lambda x : int(round(datetime.strptime(x, '%Y-%m-%d %H:%M:%S').timestamp())))


    # 2. Convert location to US-Other
    # us_url = "https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States"
    # us_states = pd.read_html(us_url)[1].iloc[:, 0].tolist()
    # dict_US_State = {k:"US" for k in us_states}

    # df["tweet_location"] = df["tweet_location"].apply(lambda x: check_US(x, dict_US_State))

    
    # 3. One hot encoding airline
    df = pd.get_dummies(df, columns=['airline'], drop_first=True)
    

    # 4. Encode airline_sentiment
    labels = ["positive", "negative", "neutral"]
    d = {k:v for k,v in zip(labels, range(3))}
    df["airline_sentiment"] = df["airline_sentiment"].apply(lambda x : d[x])


    # 5. Prepocessing text
    df["text"] = clean_text(df)

    return df


def main(params): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_bert', type=int, default=False, help='Use BERT features')
    args = parser.parse_args(params)

    use_bert = bool(args.use_bert)
    df = pre_processing()

    if use_bert: 
        print("Use BERT features")
        bert_out_df = pd.read_csv("data/bert_out.csv", index_col= 0)
        df = pd.concat([df, bert_out_df], axis = 1)



    print(df.shape)
    np.random.seed(42)
    # Split train, validation and test
    idxs = np.arange(df.shape[0])
    np.random.shuffle(idxs)

    train_size = int(len(df)*0.6) 
    val_size = int(len(df)*0.2)
    
    #train_size + val_size 
    train_idxs, val_idxs, test_idxs = idxs[:train_size], idxs[train_size:(train_size + val_size)], idxs[(train_size + val_size):]
    
    y_train = df.iloc[train_idxs]["airline_sentiment"]
    X_train = df.iloc[train_idxs].drop(columns=["airline_sentiment"])


    y_val = df.iloc[val_idxs]["airline_sentiment"]
    X_val = df.iloc[val_idxs].drop(columns=["airline_sentiment"])
    
    y_test = df.iloc[test_idxs]["airline_sentiment"]
    X_test = df.iloc[test_idxs].drop(columns=["airline_sentiment"])
    

    # Apply TF-IDF
    X_train_test, X_val_test = text_tf_idf(X_train= X_train, X_test = X_val, min_df=0.03)

    
    if use_bert == False: 
        # Apply FastTextAI 
        X_train_test, X_val_test = add_word_embeddings(X_train_test, y_train, X_val_test)

    # R
    #X_train, y_train = resampling(X_train, y_train)

    # X_train_1 = X_train.drop(columns=["text"])
    # X_val = X_val.drop(columns=["text"])

    X_train_test = X_train_test.drop(columns=["text"])
    X_val_test = X_val_test.drop(columns=["text"])
    
    results = test_models(X_train_test, y_train, X_val_test, y_val)


    print("Start tuning")

    clf_1 = RandomForestClassifier()
    best_config_RF = tuning_classifiers(clf_1, PARAMETERS_TREES, X_train_test, y_train, X_val_test, y_val)

    clf_2 = MLPClassifier()
    best_config_DT = tuning_classifiers(clf_2, PARAMETERS_MPL, X_train_test, y_train,X_val_test, y_val)

    if use_bert:
        print("BERT is not the best pretrained model so I decide not to choose it.")
    else:

        # Train on 80% of the data and then test on the test dataset 
        X = pd.concat([X_train, X_val], axis=0) 
        y = pd.concat([y_train, y_val], axis=0)

        X, X_test = text_tf_idf(X_train= X, X_test= X_test, min_df=0.03)
        X, X_test = add_word_embeddings(X, y, X_test)

        X = X.drop(columns=["text"])
        X_test = X_test.drop(columns=["text"])

        clf_1 = RandomForestClassifier(**best_config_RF)
        clf_2 = MLPClassifier(**best_config_DT)

        clf_1.fit(X, y)
        y_pred_RF = clf_1.predict(X_test)
        f1_RF = f1_score(y_test, y_pred_RF, average="macro")
        print(f"F1 score Random Forest = {f1_RF:.2f}")

        clf_2.fit(X, y)
        y_pred_DT = clf_2.predict(X_test)
        f1_DT = f1_score(y_test, y_pred_DT, average="macro")
        print(f"F1 score Multi Layer Perceptron = {f1_DT:.2f}")
    

if __name__ == '__main__':
    params = ['--use_bert','0']
    main(params)