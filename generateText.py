import pickle, torch, os
from languageModel import LSTMlanguageModel

# Use cuda if it's available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def load_vocab(path):
    with open(path, 'rb') as inf:
        vocab = pickle.load(inf)
    return vocab

def generate(model, inv_vocab, prev_word_idx, length=100):

    # "result" is a list copy of prev_word_idx, since prev_word_idx will be
    # cast into a tensor
    result = prev_word_idx
    prev_word_idx = torch.tensor(prev_word_idx).to(device)
    hidden = model.init_hidden_no_batch()


    if len(prev_word_idx) != 1:
    # Get the hidden state right before the last token
        _, hidden = model.forwardSingleSequence(prev_word_idx[:-1], hidden)

    # Prediction
    for _ in range(length - len(prev_word_idx)):
        out, hidden = model.forwardSingleSequence(torch.tensor([result[-1]]).to(device), hidden)
        multiNom = torch.exp(out)
        newword_idx = torch.multinomial(multiNom, 1).item()
        # predicted token shouldn't be <PAD> or <S>
        if newword_idx == 0 or newword_idx == 1:
            continue
        result.append(newword_idx)
        # Break if next token is end symbol </S>
        if newword_idx == 2:
            break
    return [inv_vocab[i] for i in result]



def main():
    modelFile = sys.argv[1]
    dictFile = sys.argv[2]


    embedSize = 256
    hiddSize = 256
    numLSTM = 2
    batchSize = 16
    word_to_idx = load_vocab(dictFile)
    vocabSize = len(word_to_idx)


    model = LSTMlanguageModel(embedSize, hiddSize, numLSTM,
                              vocabSize, batchSize)

    model.load_state_dict(torch.load(modelFile))
    model = model.eval()
    idx_to_word = {v: k for k, v in word_to_idx.items()}

    for i in range(5):
        print(generate(model, idx_to_word, [1], 10))









if __name__ == "__main__":
    main()
