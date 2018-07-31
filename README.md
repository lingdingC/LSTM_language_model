# LSTM_language_model
Simple feed forward language model based on LSTM

## Train
To train the model, use:
```
python(3) trainModel.py [path of preprocessed docs] [path of dict to be saved] [directory to save the models and logs]
```
It will save the vocabulary as a dictionary in the provided path as pickle file. The dictionary contains {word: word_idx} pairs.




## generate
To genrate text, use:
```
python(3) generateText.py [path of the model] [path of vocabulary dict]
```
