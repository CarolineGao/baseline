import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_data = [i for i in range(100)]  # Example input
dataset = LoRADataset(input_data, seq_len=10)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

import torch
from torch.utils.data import Dataset

class LoRADataset(Dataset):
    def __init__(self, input, seq_len):
        self.input = input
        self.seq_len = seq_len

    def __getitem__(self, item):
        return self.input[item: item + self.seq_len], self.input[item + self.seq_len]

    def __len__(self):
        return len(self.input) - self.seq_len


for sequences, targets in data_loader:
    print(sequences, targets)


# Hyper-parameters 
# input_size = 784 # 28x28
class VQAModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(VQAModel, self).__init__()
        
        # LSTM for question processing
        self.question_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # LSTM for combined image and question processing
        self.combined_lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer for answer prediction
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, image, question):
        # Flatten image
        image_seq = image.view(image.size(0), -1, input_size)
        
        # Pass question through LSTM
        h0 = torch.zeros(self.num_layers, question.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, question.size(0), self.hidden_size).to(device)
        question_out, _ = self.question_lstm(question, (h0, c0))
        question_out = question_out[:, -1, :]
        
        # Concatenate image sequence and question representation
        combined_seq = torch.cat((image_seq, question_out.unsqueeze(1).repeat(1, image_seq.size(1), 1)), 1)
        
        # Pass combined sequence through LSTM
        h0_combined = torch.zeros(self.num_layers, combined_seq.size(0), self.hidden_size).to(device)
        c0_combined = torch.zeros(self.num_layers, combined_seq.size(0), self.hidden_size).to(device)
        lstm_out, _ = self.combined_lstm(combined_seq, (h0_combined, c0_combined))
        lstm_out = lstm_out[:, -1, :]
        
        # Get predictions
        out = self.fc(lstm_out)
        return out
