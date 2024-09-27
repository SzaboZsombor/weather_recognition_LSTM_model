import torch
import torch.nn as nn


class LSTM_weather(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, seq_length, output_length):
        super(LSTM_weather, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.output_length = output_length

        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, self.hidden_size).to(x.device)

        output, (hn, cn) = self.lstm(x, (h_0, c_0))

        out = output[-self.output_length:, :]

        out = self.relu(self.fc(out))

        return out


def predict(input_df):
    model_path = r"D:\Machine Learning\weather_prediction\best_model2.pth"

    input_df = torch.tensor(input_df, dtype=torch.float32).to('cpu')

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    input_dim = checkpoint['input_dim']
    hidden_size = checkpoint['hidden_size']
    num_layers = checkpoint['num_layers']
    seq_length = checkpoint['seq_length']
    output_length = checkpoint['output_length']

    model = LSTM_weather(input_dim, hidden_size, num_layers, seq_length, output_length)

    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    with torch.no_grad():
        pred_temp = model(input_df)

    return pred_temp.numpy()
