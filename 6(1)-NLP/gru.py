import torch
from torch import nn, Tensor


class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Update gate
        self.W_z = nn.Linear(input_size, hidden_size, bias=False)
        self.U_z = nn.Linear(hidden_size, hidden_size)

        # Reset gate
        self.W_r = nn.Linear(input_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size)

        # Candidate hidden state
        self.W_h = nn.Linear(input_size, hidden_size, bias=False)
        self.U_h = nn.Linear(hidden_size, hidden_size)


    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        z = torch.sigmoid(self.W_z(x) + self.U_z(h))       # update gate
        r = torch.sigmoid(self.W_r(x) + self.U_r(h))       # reset gate
        h_tilde = torch.tanh(self.W_h(x) + self.U_h(r * h))  # candidate
        h_new = (1 - z) * h + z * h_tilde                  # final hidden state
        return h_new


class GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = GRUCell(input_size, hidden_size)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        inputs: Tensor of shape (batch_size, seq_len, input_size)
        returns: Tensor of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = inputs.size()
        h = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        for t in range(seq_len):
            x_t = inputs[:, t, :]  # (batch_size, input_size)
            h = self.cell(x_t, h)

        return h # hidden state