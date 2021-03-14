import torch
import math
import numpy as np

class FeedForwardEmbedNN(torch.nn.Module):


    def __init__(self, n_users, n_movies, hidden, dropouts, n_factors, embedding_dropout):
        super().__init__()
        self.user_emb = torch.nn.Embedding(n_users, n_factors)
        self.movie_emb = torch.nn.Embedding(n_movies, n_factors)
        self.drop = torch.nn.Dropout(embedding_dropout)
        self.hidden_layers = torch.nn.Sequential(*list(self.generate_layers(n_factors*2, hidden, dropouts)))
        self.fc = torch.nn.Linear(hidden[-1], 1)


    def generate_layers(self, n_factors, hidden, dropouts):
        assert len(dropouts) == len(hidden)

        idx = 0
        while idx < len(hidden):
            if idx == 0:
                yield torch.nn.Linear(n_factors, hidden[idx])
            else:
                yield torch.nn.Linear(hidden[idx-1], hidden[idx])
            yield torch.nn.ReLU()
            yield torch.nn.Dropout(dropouts[idx])

            idx += 1


    def forward(self, users, movies, min_rating=0.5, max_rating=5):
        concat_features = torch.cat([self.user_emb(users), self.movie_emb(movies)], dim=1)
        x = torch.nn.functional.relu(self.hidden_layers(concat_features))
        # 0과 1사이의 숫자로 나타낸다
        out = torch.sigmoid(self.fc(x))
        # rating으로 변환한다
        out = (out * (max_rating - min_rating)) + min_rating

        return out

    def predict(self, users, movies):
        # return the score
        output_scores = self.forward(users, movies)
        return output_scores


class BatchIterator:

    def __init__(self, X, y, batch_size=32, shuffle=True):
        X, y = np.asarray(X), np.asarray(y)

        if shuffle:
            index = np.random.permutation(X.shape[0])
            X, y = X[index], y[index]

        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_batches = int(math.ceil(X.shape[0] // batch_size))
        self._current = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self._current >= self.n_batches:
            raise StopIteration()
        k = self._current
        self._current += 1
        bs = self.batch_size
        return self.X[k * bs:(k + 1) * bs], self.y[k * bs:(k + 1) * bs]


def batches(X, y, bs=32, shuffle=True):
    for x_batch, y_batch in BatchIterator(X, y, bs, shuffle):
        x_batch = torch.LongTensor(x_batch)
        y_batch = torch.FloatTensor(y_batch)
        yield x_batch, y_batch.view(-1, 1)