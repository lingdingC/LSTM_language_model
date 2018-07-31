import torch
from torch.utils.data import Dataset
from collections import Counter

class TextData(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)



def collate_fn(data):
    """
    Pad the documents in a batch to the same length
    data: list of torch tensor of documents as word_idx
          e.g. [[doc_tensor0], [doc_tensor1], ...]
    """
    # Sort this so that doc of similar length will go
    data.sort(key=len, reverse=True)

    docLengths = [len(doc) for doc in data]

    paddedDocs = torch.zeros(len(data), max(docLengths)).long()
    for idx, doc in enumerate(data):
        paddedDocs[idx, :docLengths[idx]] = data[idx]

    return paddedDocs, docLengths


def readFile(fileName, delim=","):
    """
    Read tokenized documents from CSV like files
    Transform tokens to word_idx
    """
    documents = []
    with open(fileName, 'rt') as csvin:
        for line in csvin:
            # remove newline char at the end and split
            doc = line[:-1].split(delim)
            documents.append(doc)
    return documents



def buildVocabDict(documents, maxVocabSize=20000):
    """
    Learned this counter() trick from Prof. Jurgens
    """
    tokenCounter = Counter()
    for doc in documents:
        for word in doc:
            tokenCounter[word] += 1
    mostFrequentWords = set([x[0] for x in tokenCounter.most_common(20000)])

    wordVocab = {}
    wordVocab['<PAD>'] = 0
    wordVocab['<S>']   = 1
    wordVocab['</S>']  = 2

    for word in mostFrequentWords:
        wordVocab[word] = len(wordVocab)

    return wordVocab


def word2idx(vocabDict, doc):
    """
    Change the vector of words into a vector of indice
    """
    return [vocabDict[word] for word in doc]
