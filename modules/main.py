import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from .data_preprocessing import TextPreprocessing


class NLP:
    def __init__(self):
        self.train_df = pd.read_csv("data/train.csv")
        self.test_df = pd.read_csv("data/test.csv")
        self.model_lr = linear_model.RidgeClassifier()
        self.model_rfc = RandomForestClassifier()

        self.OHE = preprocessing.OneHotEncoder()

    def main(self):
        count_vectorizer = feature_extraction.text.CountVectorizer()

        self.train_df = self.data_preprocessing(self.train_df)
        self.test_df = self.data_preprocessing(self.test_df)

        train_vectors = count_vectorizer.fit_transform(self.train_df["text"])
        test_vectors = count_vectorizer.transform(self.test_df["text"])

        # self.train_lr(train_vectors, test_vectors)
        self.train_rfc(train_vectors, test_vectors)

    @staticmethod
    def data_preprocessing(df):
        text_preprocessing = TextPreprocessing(df)
        return text_preprocessing.main()

    def train_lr(self, train_vectors, test_vectors):
        scores = model_selection.cross_val_score(self.model_lr, train_vectors, self.train_df["target"], cv=3, scoring="f1")
        print(f'LR CV scores: {scores}')

        self.model_lr.fit(train_vectors, self.train_df["target"])

        sample_submission = self.to_predict_lr(test_vectors)
        self.save_result(sample_submission)

    def to_predict_lr(self, test_vectors):
        sample_submission = pd.read_csv("data/sample_submission.csv")
        sample_submission["target"] = self.model_lr.predict(test_vectors)
        return sample_submission

    def train_rfc(self, train_vectors, test_vectors):
        B = pd.concat([self.train_df[['location', 'keyword']], self.test_df[['location', 'keyword']]])
        X_cat = self.OHE.fit_transform(B)

        y = self.train_df['target'].to_numpy()
        X_cat = self.OHE.transform(self.train_df[['location', 'keyword']])
        X = hstack((X_cat, train_vectors))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        self.model_rfc.fit(X_train, y_train)
        score = self.model_rfc.score(X_test, y_test)
        print(f'RFC score: {score}')

        sample_submission = self.to_predict_rfc(test_vectors)
        self.save_result(sample_submission, model='rfc')

    def to_predict_rfc(self, test_vectors):
        sample_submission = pd.read_csv("data/sample_submission.csv")
        X_cat = self.OHE.transform(self.test_df[['location', 'keyword']])
        X = hstack((X_cat, test_vectors))
        sample_submission["target"] = self.model_rfc.predict(X)
        return sample_submission

    @staticmethod
    def save_result(sample_submission, model='lr'):
        name = "submission_" + model + " .csv"
        sample_submission.to_csv(name, index=False)





