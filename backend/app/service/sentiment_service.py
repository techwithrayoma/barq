# from transformers import pipeline


class SentimentService:
    def __init__(self, model_name="tabularisai/multilingual-sentiment-analysis"):
        # self.analyzer = pipeline(
        #     "sentiment-analysis",
        #     model=model_name,
        #     device=-1,
        #     truncation=True,
        #     max_length=512
        # )
        pass

    def predict(self, comment: str):
        # result = self.analyzer(comment)
        # label = result[0]['label'].lower()
        # if 'negative' in label else 'positive'
        return 'negative' 
