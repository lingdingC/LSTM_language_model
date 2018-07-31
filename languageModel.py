import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Use cuda if is_available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")



class LSTMlanguageModel(nn.Module):
    def __init__(self, embeddingDim, hiddenDim, numLSTMLayer, vocabSize,
                 batchSize, dropout=0.2):

        super(LSTMlanguageModel, self).__init__()

        self.hiddenDim = hiddenDim
        self.embeddingDim = embeddingDim
        self.numLSTMLayer = numLSTMLayer
        self.vocabSize = vocabSize
        self.batchSize = batchSize

        self.embedding = nn.Embedding(vocabSize, embeddingDim).to(device)
        self.lstm = nn.LSTM(embeddingDim, hiddenDim,
                            num_layers=numLSTMLayer, dropout=dropout).to(device)
        self.decoder = nn.Linear(hiddenDim, vocabSize).to(device)

    def init_hidden(self):
        return (torch.zeros(self.numLSTMLayer, self.batchSize, self.hiddenDim),
                torch.zeros(self.numLSTMLayer, self.batchSize, self.hiddenDim))


    def init_hidden_no_batch(self):
        """
        Used for predicting
        """
        return (torch.zeros(self.numLSTMLayer, 1, self.hiddenDim),
                torch.zeros(self.numLSTMLayer, 1, self.hiddenDim))


    #TODO: implement customized weight init function

    def forward(self, input, hidden, lengths):
        """
        Input is a vector of token_idx
        This forward function is used for training with batches
        Use forwardSingleSequence for predicting
        """

        embeds = self.embedding(input)
        # Pack the padded sequence; maxLength found by DataLoader


        padded_embeds = pack_padded_sequence(embeds, lengths, batch_first=True)

        lstmOut, hidden = self.lstm(padded_embeds, hidden)
        # Unpack
        lstmOut, _ = pad_packed_sequence(lstmOut, batch_first=True)

        vocab_space = self.decoder(lstmOut)
        return vocab_space, hidden


    def forwardSingleSequence(self, input, hidden):
        """
        Input is a vector of word_idx
        """
        embeds = self.embedding(input)
        lstmOut, self.hidden = self.lstm(embeds.view(len(input), 1, -1), hidden)
        vocab_space = self.decoder(lstmOut.view(len(input), -1))
        return F.log_softmax(vocab_space, dim=1), hidden
