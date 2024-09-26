# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.). You will need to implement the predict
    method and you can optionally override predict_all if you want to use batching at inference time (not necessary,
    but may make things faster!)
    """
#     def __init__(self):
#         raise NotImplementedError
    def __init__(self, model, word_embeddings: WordEmbeddings):
        super(NeuralSentimentClassifier, self).__init__()
        self.model = model
        self.word_embeddings = word_embeddings

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        self.model.eval()  # Set the model to evaluation mode
        # if has_typos:
        #     ex_words = [word[:3] for word in ex_words]

        with torch.no_grad():  # No need to track gradients during inference
            # Use pad_and_average_word_embeddings to get the average embedding for the sentence
            avg_embedding = pad_and_average_word_embeddings([ex_words], self.word_embeddings, self.model.fc1.in_features)
            avg_embedding = avg_embedding.to(next(self.model.parameters()).device)  # Ensure it is on the same device
            output = self.model(avg_embedding)
            _, predicted = torch.max(output, 1)
        return predicted.item()
    
class DeepAveragingNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.1):
        super(DeepAveragingNetwork, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x = self.dropout(x)
        x = self.relu1(self.fc1(x))
        # x = self.dropout(x)
        x = self.relu2(self.fc2(x))
        # x = self.dropout(x)
        x = self.fc3(x)
        return x


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :param train_model_for_typo_setting: True if we should train the model for the typo setting, False otherwise
    :return: A trained NeuralSentimentClassifier model. Note: you can create an additional subclass of SentimentClassifier
    and return an instance of that for the typo setting if you want; you're allowed to return two different model types
    for the two settings.
    """

    if train_model_for_typo_setting:
        word_embeddings = PrefixEmbeddings(word_embeddings, prefix_length=3)
    embedding_dim = 300 # 50 if chage to the other link
    model = DeepAveragingNetwork(input_dim=embedding_dim, hidden_dim=args.hidden_size, output_dim=2)
    batch_size = args.batch_size
    num_epochs = 10 #args.num_epochs

    # define criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-3)
   
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model.to(device)
    initialize_weights(model)

    # training
    for epoch in range (num_epochs):
        model.train()
        total_loss = 0
        random.shuffle(train_exs)
        for i in range(0, len(train_exs), batch_size):
            batch_sentences = [ex.words for ex in train_exs[i:i+batch_size]]
            batch_labels = torch.tensor([ex.label for ex in train_exs[i:i+batch_size]])
            avg_embeddings = pad_and_average_word_embeddings(batch_sentences, word_embeddings, embedding_dim)
            outputs = model(avg_embeddings)
            loss = criterion(outputs, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        dev_accuracy = evaluate(model, dev_exs, word_embeddings, embedding_dim, batch_size, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_exs):.4f}, Dev Accuracy: {dev_accuracy * 100:.2f}%")
    return NeuralSentimentClassifier(model, word_embeddings)

import torch.nn.init as init

def initialize_weights(model):
    """
    init function
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):  
            init.xavier_uniform_(m.weight)  
            if m.bias is not None:
                init.zeros_(m.bias)  


def pad_and_average_word_embeddings(batch_sentences, word_embeddings, embedding_dim, padding_idx=0):
    """
    :param batch_sentences: List of sentences (each sentence is a list of words)
    :param word_embeddings: WordEmbeddings object that provides embeddings
    :param embedding_dim: Dimension of word embeddings
    :param padding_idx: Index of the PAD token
    :return: Tensor of shape (batch_size, embedding_dim)
    """
    batch_size = len(batch_sentences)
    max_len = max(len(sentence) for sentence in batch_sentences)

    embeddings = torch.zeros((batch_size, max_len, embedding_dim))
    mask = torch.zeros((batch_size, max_len))

    # init and set padding idx to 0
    embeddings = torch.zeros((batch_size, max_len, embedding_dim))

    for i, sentence in enumerate(batch_sentences):
        sentence_embeddings = [word_embeddings.get_embedding(word) for word in sentence]

        #  numpy.ndarray convert to torch.Tensor
        sentence_embeddings = [torch.tensor(embedding) if isinstance(embedding, np.ndarray) else embedding for embedding in sentence_embeddings]
        sentence_embeddings = torch.stack(sentence_embeddings)
        if len(sentence) < max_len:
            padding = torch.zeros((max_len - len(sentence), embedding_dim))
            sentence_embeddings = torch.cat((sentence_embeddings, padding), dim=0)
        mask[i, :len(sentence)] = 1
        embeddings[i] = sentence_embeddings
    # avg_embeddings = embeddings.mean(dim=1)
    mask = mask.unsqueeze(2)  # [batch_size, max_len, 1] for broadcast
    masked_embeddings = embeddings * mask 
    # sum embeddings along the time dimension (ignoring padding)
    sum_embeddings = masked_embeddings.sum(dim=1)
    # count non-padding tokens for each sentence and use it to divide the sum
    lengths = mask.sum(dim=1)  # [batch_size, 1] 
    lengths = torch.clamp(lengths, min=1)
    avg_embeddings = sum_embeddings / lengths
    
    return avg_embeddings

import torch

def evaluate(model, dev_exs, word_embeddings, embedding_dim, batch_size, device):
    model.eval()  
    correct = 0
    total = 0

    with torch.no_grad():  
        for i in range(0, len(dev_exs), batch_size):
            batch_sentences = [ex.words for ex in dev_exs[i:i+batch_size]]
            batch_labels = torch.tensor([ex.label for ex in dev_exs[i:i+batch_size]]).to(device)
            avg_embeddings = pad_and_average_word_embeddings(batch_sentences, word_embeddings, embedding_dim).to(device)
            outputs = model(avg_embeddings)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_labels).sum().item()
            total += batch_labels.size(0)

    accuracy = correct / total
    return accuracy

class PrefixEmbeddings:
    def __init__(self, word_embeddings, prefix_length=3):
        """
        :param word_embeddings: WordEmbeddings 
        :param prefix_length: default 3
        """
        self.prefix_length = prefix_length
        self.prefix_to_idx = {}  
        self.embeddings = None  
        self.word_indexer = word_embeddings.word_indexer
        self.vectors = word_embeddings.vectors
        self._initialize_prefix_embeddings(word_embeddings)

    def _initialize_prefix_embeddings(self, word_embeddings):
        prefix_to_embeddings = {}
        vocab = word_embeddings.word_indexer  

        # iterate all words and get the prefix, and use the prefix for avg embedding
        for i in range(len(vocab)):
            word = vocab.get_object(i)  
            embedding = word_embeddings.get_embedding(word)  
            prefix = word[:self.prefix_length] 
            if prefix not in prefix_to_embeddings:
                prefix_to_embeddings[prefix] = []
            prefix_to_embeddings[prefix].append(torch.tensor(embedding))  

        # cal avg embeddings for prefix
        for prefix, embeddings in prefix_to_embeddings.items():
            avg_embedding = torch.mean(torch.stack(embeddings), dim=0)
            self.prefix_to_idx[prefix] = len(self.prefix_to_idx)  # assign each prefix an index
            if self.embeddings is None:
                self.embeddings = avg_embedding.unsqueeze(0)
            else:
                self.embeddings = torch.cat((self.embeddings, avg_embedding.unsqueeze(0)), dim=0)

        self.embeddings = nn.Parameter(self.embeddings)  

    def get_embedding(self, word: str) -> torch.Tensor:
        prefix = word[:self.prefix_length]  
        if len(prefix) < self.prefix_length:
            prefix = word  

        if prefix in self.prefix_to_idx:
            idx = self.prefix_to_idx[prefix]
            return self.embeddings[idx]
        else:
        # If prefix is not found, fall back to word_indexer to check the word
            word_idx = self.word_indexer.index_of(word)
            if word_idx != -1:
                return self.vectors[word_idx]
            else:
            # If the word is also not found, return the vector for "UNK"
                return self.vectors[self.word_indexer.index_of("UNK")]
