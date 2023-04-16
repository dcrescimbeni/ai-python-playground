# Sentiment Analysis with Naive Bayes Classifier

This Python code performs sentiment analysis on text data using a Multinomial Naive Bayes classifier. It is a test project aimed at getting familiarized with AI.

## Data Source

The data used in this project can be found [here](https://www.kaggle.com/datasets/kazanova/sentiment140).

## How to Run

1. Run `pipenv install` to install the required Python packages: pandas, scikit-learn
2. Download the data from the provided link and place it in the same directory as the Python script under the name `tweet-sentiment.csv`.
3. Run the script in a Python environment: `python3 sentiment-prediction.py`.

## Output

The script allows for user input to analyze the sentiment of a tweet. The output is either "positive" or "negative".

## Note

The model evaluation stats have been commented out in the code for simplicity. Uncomment them to evaluate the performance of the model.
