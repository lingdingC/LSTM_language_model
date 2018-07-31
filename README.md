# LSTM_language_model
Simple feed forward language model based on LSTM

To train the model, use:
    python(3) trainModel.py [path of preprocessed documents] [path of dict to be saved] [directory to save the models]
    
To genrate text, use:
    python(3) generateText.py [path of the model] [path of vocabulary dict]
