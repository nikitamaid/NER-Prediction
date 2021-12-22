from dataset_preparation import build_corpus, build_corpus_test, build_map_tags, build_map_words, sentence_vectorizer, \
    tag_vectorizer
from model2 import BiLSTM
import pandas as pd
from dataLoader import MyDataLoader, MyDataLoader_fortest, MyCollator_fortest, MyCollator
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np

train_words, train_tags = build_corpus('train')
dev_word_lists, dev_tag_lists = build_corpus("dev")
test_word_lists = build_corpus_test('test')
word2id = build_map_words(train_words, dev_word_lists, test_word_lists)
tag2id = build_map_tags(train_tags, dev_tag_lists)
print('Training data', len(train_words), len(train_tags))
print('Validation data', len(dev_word_lists), len(dev_tag_lists))
print('Testing data', len(test_word_lists))

train_word_vector = sentence_vectorizer(train_words, word2id)
test_word_vector = sentence_vectorizer(test_word_lists, word2id)
dev_word_vector = sentence_vectorizer(dev_word_lists, word2id)
train_tag_vector = tag_vectorizer(train_tags, tag2id)
dev_tag_vector = tag_vectorizer(dev_tag_lists, tag2id)


def create_emb_matrix(word2id, emb_dict, dimension):
    emb_matrix = np.zeros((len(word2id), dimension))
    for word, idx in word2id.items():
        if word in emb_dict:
            emb_matrix[idx] = emb_dict[word]
        else:
            if word.lower() in emb_dict:
                emb_matrix[idx] = emb_dict[word.lower()] + 5e-3
            else:
                emb_matrix[idx] = emb_dict["<UNK>"]

    return emb_matrix


glove = pd.read_csv('glove.6B.100d.gz', sep=" ",
                    quoting=3, header=None, index_col=0)
glove_emb = {key: val.values for key, val in glove.T.items()}

glove_vec = np.array([glove_emb[key] for key in glove_emb])
glove_emb["<PAD>"] = np.zeros((100,), dtype="float64")
glove_emb["<UNK>"] = np.mean(glove_vec, axis=0, keepdims=True).reshape(100, )
embeddingMatrix = create_emb_matrix(
    word2id=word2id, emb_dict=glove_emb, dimension=100)

vocab_size = embeddingMatrix.shape[0]
vector_size = embeddingMatrix.shape[1]
print(vocab_size, vector_size)

BiLSTM_model = BiLSTM(vocab_size=len(word2id),
                      embedding_dim=100,
                      linear_out_dim=128,
                      hidden_dim=256,
                      lstm_layers=1,
                      bidirectional=True,
                      dropout_val=0.33,
                      tag_size=len(tag2id),
                      embeddingMatrix=embeddingMatrix)

print(BiLSTM_model)

training_data = MyDataLoader(train_word_vector, train_tag_vector)
collator_fn = MyCollator(word2id, tag2id)
dataloader = DataLoader(dataset=training_data,
                        batch_size=8,
                        drop_last=True,
                        collate_fn=collator_fn)


criterion = nn.CrossEntropyLoss()
criterion.requres_grad = True
optimizer = torch.optim.SGD(BiLSTM_model.parameters(), lr=0.1, momentum=0.9)
epochs = 10

for i in range(1, epochs+1):
    train_loss = 0.0
    # scheduler.step(train_loss)
    for input, label, input_len, label_len in dataloader:
        optimizer.zero_grad()
        output = BiLSTM_model(input, input_len)  # input_len
        output = output.view(-1, len(tag2id))
        label = label.view(-1)
        loss = criterion(output, label)
        # print(loss)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * input.size(1)

    train_loss = train_loss / len(dataloader.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(i, train_loss))
torch.save(BiLSTM_model.state_dict(),'blstm2.pt')
BiLSTM_dev = MyDataLoader(dev_word_vector, dev_tag_vector)
collator_fn = MyCollator(word2id, tag2id)
dataloader_dev = DataLoader(dataset=BiLSTM_dev,
                            batch_size=1,
                            shuffle=False,
                            drop_last=True,
                            collate_fn=collator_fn)

id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
id2word = dict((id_, word) for word, id_ in word2id.items())

dev_pred = []
for dev_data, label, dev_data_len, label_data_len in dataloader_dev:

    pred = BiLSTM_model(dev_data, dev_data_len)
    pred = pred.cpu()
    pred = pred.detach().numpy()
    label = label.detach().numpy()
    dev_data = dev_data.detach().numpy()
    pred = np.argmax(pred, axis=2)
    pred = pred.reshape((len(label), -1))

    for i in range(len(dev_data)):
        for j in range(len(dev_data[i])):
            if dev_data[i][j] != 0:
                word = id2word[dev_data[i][j]]
                gold = id2tag[label[i][j]]
                op = id2tag[pred[i][j]]
                dev_pred.append((word, gold, op))

with open("pred2.txt", 'w') as f:
    idx = 1
    for ele in dev_pred:
        f.write(" ".join([str(idx), ele[0], ele[1], ele[2]]))
        f.write("\n")
        idx += 1

print('creating dev2.out')
with open("dev2.out", 'w') as f:
    i = 1
    for ele in dev_pred:
        if ele[0] != '.':
            f.write(" ".join([str(i), ele[0], ele[2]]))
            f.write("\n")
            i = i + 1
        else:
            f.write(" ".join([str(i), ele[0], ele[2]]))
            f.write("\n")
            f.write("\n")
            i = 1


testing_data = MyDataLoader_fortest(test_word_vector)
collator_fn1 = MyCollator_fortest(word2id, tag2id)
dataloader_test = DataLoader(dataset=testing_data,
                             batch_size=1,
                             shuffle=False,
                             drop_last=True,
                             collate_fn=collator_fn1)

test_pred = []
for dev_data , dev_data_len in dataloader_test:
    pred = BiLSTM_model(dev_data, dev_data_len)
    pred = pred.cpu()
    pred = pred.detach().numpy()
    dev_data = dev_data.detach().numpy()
    pred = np.argmax(pred, axis=2)
    #     pred = pred.reshape((len(label), -1))

    for i in range(len(dev_data)):
        for j in range(len(dev_data[i])):
            if dev_data[i][j] != 0:
                word = id2word[dev_data[i][j]]
                op = id2tag[pred[i][j]]
                test_pred.append((word, op))

print('creating test2.out')
with open("test2.out", 'w') as f:
    i = 1
    for ele in test_pred:
        if ele[0] != '.':
            f.write(" ".join([str(i), ele[0], ele[1]]))
            f.write("\n")
            i = i + 1
        else:
            f.write(" ".join([str(i), ele[0], ele[1]]))
            f.write("\n")
            f.write("\n")
            i = 1

print("End of task 2")