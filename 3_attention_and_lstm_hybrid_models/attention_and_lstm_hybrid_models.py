import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define constants
feature = 8
timestep = 4
batch_size = 32  # Example batch size
epochs = 100  # Example number of epochs
learning_rate = 0.001  # Learning rate


# Define attention mechanisms and LSTM models (as in the original code)
class Feature_attention(nn.Module):
    def __init__(self, d):
        super(Feature_attention, self).__init__()
        self.fn = nn.Linear(d, d)
        self.scale_factor = np.sqrt(d)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, enc_inputs):
        outputs = self.fn(enc_inputs)
        outputs = self.sigmoid(outputs)
        attn = self.softmax(outputs)
        outputs = torch.mul(enc_inputs, attn)
        return outputs, attn


class Temporal_attention(nn.Module):
    def __init__(self, d):
        super(Temporal_attention, self).__init__()
        self.fn = nn.Linear(d, d)
        self.scale_factor = np.sqrt(d)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, enc_inputs):
        outputs = self.fn(enc_inputs)
        outputs = self.sigmoid(outputs)
        attn = self.softmax(outputs)
        outputs = torch.mul(enc_inputs, attn)
        return outputs, attn


class FA_lstm(nn.Module):
    def __init__(self, d1):
        super(FA_lstm, self).__init__()
        self.feature_Attn = Feature_attention(d1)
        self.lstm = nn.LSTM(feature, 16, 2)
        self.fc = nn.Linear(16, 1)

    def forward(self, inputs):
        self.lstm.flatten_parameters()
        spa_outputs, spa_attn = self.feature_Attn(inputs)
        spa_outputs = spa_outputs  # apply feature attention
        lstm_outputs = self.lstm(spa_outputs.permute(1, 0, 2))[0]
        lstm_outputs = self.fc(lstm_outputs)
        lstm_outputs = lstm_outputs.permute(1, 0, 2)
        return lstm_outputs[:, timestep - 1, :]


# Additional models (TA_lstm, FTA_lstm) remain the same

# Training the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, targets)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            running_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}')


# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    print(f'Test Loss: {total_loss / len(test_loader)}')


# Example dataset and DataLoader (replace with actual data)
# Assuming time series data with dimensions [batch_size, timestep, feature]
train_data = torch.randn(batch_size, timestep, feature)
train_targets = torch.randn(batch_size, 1)  # Example target data

# Example DataLoader
train_loader = [(train_data, train_targets)]  # Use actual DataLoader in practice
test_loader = [(train_data, train_targets)]  # Use actual test data

# Initialize model, criterion, optimizer
model = FA_lstm(feature)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train and evaluate the model
train_model(model, train_loader, criterion, optimizer, num_epochs=epochs)
evaluate_model(model, test_loader)
