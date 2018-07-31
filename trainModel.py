import pickle, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from loadData import TextData, collate_fn, readFile, buildVocabDict, word2idx
from languageModel import LSTMlanguageModel

# Use cuda if is_available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def repackage_hidden(h):
    """This is from pyTorch example"""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def trainModel(model, documents, vocab, batch_size, modelName="",
                num_iterations=1000):


    trainingSet = [torch.tensor(word2idx(vocab, doc)) for doc in documents]

    tag_pad_token = vocab['<PAD>']

    loss_function = nn.CrossEntropyLoss(ignore_index=tag_pad_token)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    numDoc = len(trainingSet)
    hidden = model.init_hidden()


    train_loader = DataLoader(dataset=TextData(trainingSet),
                              batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=1, pin_memory=False,
                              collate_fn=collate_fn)

    print('loaded data')

    for epoch in range(num_iterations):

        totalLoss = 0

        for bi, trainBatch in enumerate(train_loader):


            model.zero_grad()

            hidden = repackage_hidden(hidden)


            batched_documents, lengths = trainBatch

            lengths = [num - 1 for num in lengths]

            doc_in = batched_documents[:,:-1].to(device)
            doc_target = batched_documents[:,1:].to(device)

            target_scores, hidden = model(doc_in, hidden, lengths)


            flattened_ts = target_scores.contiguous().view(target_scores.size(1)*batch_size, -1)
            flattened_target = doc_target.contiguous().view(1, doc_target.size(1)*batch_size).squeeze(0)


            loss = loss_function(flattened_ts, flattened_target)

            # use .item() so that the tensor doesn't grow and blowup memory
            totalLoss += loss.item()
            loss.backward()
            optimizer.step()


        # Change this to match actual training data size
        if epoch > 0 and epoch % 100 == 0:
            print(str(epoch) + " epoch")
            print("loss: " + str(totalLoss / numDoc))
            with open("./test/trainingLog/testTrainingLog.txt", "a") as outf:
                outf.write("{0}\t{1}\n".format(epoch, totalLoss / numDoc))
            save_model(model, modelName + "e" + str(epoch))

    print("finish training")
    return model


def save_model(model, model_name, vocab=None):

    torch.save(model.state_dict(), "./test/models/{0}_hid{1}_emb{2}_vocab{3}".format(
        model_name, model.hiddenDim, model.embeddingDim,
        model.vocabSize))

    if vocab is not None:
        with open("./test/vocabulary/{0}_hid{1}_emb{2}_vocab{3}".format(
                model_name, model.hiddenDim, model.embeddingDim,
                model.vocabSize), "wb") as outf:
            pickle.dump(vocab, outf)



def main():

    samplePath = "./test/documents/testDoc.csv"
    text_documents = readFile(samplePath)
    word_to_idx = buildVocabDict(text_documents)
    print(word_to_idx)

    dictPath = "./test/vocabulary/test_word_to_idx"
    # Save the dict so that it can be used for predicting
    with open(dictPath, 'wb') as outf:
        pickle.dump(word_to_idx, outf)

    # Customize parameters
    embedSize = 256
    hiddSize = 256
    numLSTM = 2
    batchSize = 16
    vocabSize = len(word_to_idx)
    print(word_to_idx)

    model = LSTMlanguageModel(embedSize, hiddSize, numLSTM,
                              vocabSize, batchSize)
    print("trainning model")
    model = trainModel(model, text_documents, word_to_idx, batchSize)



if __name__ == "__main__":
    main()
