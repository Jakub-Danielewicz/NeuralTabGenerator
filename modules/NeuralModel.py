from torch.utils.data import Dataset,DataLoader, SubsetRandomSampler
import torch
import torch.nn as nn
import numpy as np


def getLabels(model, data):
    SongLoader = DataLoader(dataset=data, batch_size=data.num_sequences, shuffle=False)

    for i, features in enumerate(SongLoader, 1):
        outputs = model(features)


    outputs = outputs.view(data.num_sequences, data.sequence_length, 6, 27)
    _, preds = torch.max(outputs, 3)
    preds = preds.view(-1, 6)

    # Czyszczenie tensora z paddingu
    features = features.view(-1, 128)
    padded = (features == 26).all(dim=1)
    padded = torch.nonzero(padded).squeeze()

    mask = torch.ones(preds.shape[0], dtype=torch.bool)
    mask[padded] = False
    return preds[mask]
class GuitarInferenceDataset(Dataset):
    def __init__(self, dataset, sequence_length):
        self.sequence_length = sequence_length

        # Identify indices where EOS tokens occur (assuming it marks the end of a song)

        # Create padded sequences
        self.sequences = []
        for idx in range(0, len(dataset), self.sequence_length):
            # Adjust the end index if it's before a song boundary
            end_idx = min(idx + self.sequence_length, len(dataset))

            sequence = dataset[idx:end_idx]

            # Pad the sequence if it's shorter than the specified length
            if len(sequence) < self.sequence_length:
                pad_length = self.sequence_length - len(sequence)
                pad_vector = np.zeros((pad_length, 128)) + 26  # Create padding vectors
                sequence = np.vstack((pad_vector, sequence))  # Vertically stack with the sequence

            self.sequences.append(sequence)

        # Update the number of sequences after zero-padding
        self.num_sequences = len(self.sequences)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        # Get the sequence based on index
        sequence = self.sequences[idx]

        # Split the sequence into features and labels
        features = torch.tensor(sequence[:, :128], dtype=torch.float32)  # First 128 values as features

        return features




class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, sequence_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_size = sequence_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.batchnorm = nn.BatchNorm1d(sequence_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        out, (h0, c0) = self.lstm(x, (h0, c0))

        out = self.batchnorm(out)
        out = self.relu(self.fc1(out))


        outputs = self.fc2(out)

        return outputs

