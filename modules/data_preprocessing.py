import re
import string


class TextPreprocessing:

    def __init__(self, df):
        self.df = df

    def main(self):
        self.process_text_custom()
        self.df['text'] = self.df['text'].apply(lambda x: self.process_text_myself(x))

        return self.df

    def process_text_custom(self):
        self.df['text'] = self.df['text'].apply(lambda x: x.lower())

    def process_text_myself(self, text):
        text = self.remove_html(text)
        text = self.remove_url(text)
        text = self.remove_emoji(text)
        text = self.remove_punctuation(text)
        return text

    @staticmethod
    def remove_url(text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'', text)

    @staticmethod
    def remove_html(text):
        html = re.compile(r'<.*?>')
        return html.sub(r'', text)

    @staticmethod
    def remove_emoji(text):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    @staticmethod
    def remove_punctuation(text):
        table = str.maketrans("", "", string.punctuation)
        return text.translate(table)
