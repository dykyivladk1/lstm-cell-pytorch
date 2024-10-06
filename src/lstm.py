import torch
import torch.nn.functional as F

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_f = torch.randn(hidden_size, input_size + hidden_size, requires_grad=True)
        self.b_f = torch.zeros(hidden_size, 1, requires_grad=True)
        self.W_i = torch.randn(hidden_size, input_size + hidden_size, requires_grad=True)
        self.b_i = torch.zeros(hidden_size, 1, requires_grad=True)
        self.W_C = torch.randn(hidden_size, input_size + hidden_size, requires_grad=True)
        self.b_C = torch.zeros(hidden_size, 1, requires_grad=True)
        self.W_o = torch.randn(hidden_size, input_size + hidden_size, requires_grad=True)
        self.b_o = torch.zeros(hidden_size, 1, requires_grad=True)

    def forward(self, x, h_prev, C_prev):
        concat = torch.cat((h_prev, x), dim=0)
        f_t = torch.sigmoid(self.W_f @ concat + self.b_f)
        i_t = torch.sigmoid(self.W_i @ concat + self.b_i)
        C_tilde = torch.tanh(self.W_C @ concat + self.b_C)
        C_t = f_t * C_prev + i_t * C_tilde
        o_t = torch.sigmoid(self.W_o @ concat + self.b_o)
        h_t = o_t * torch.tanh(C_t)
        return h_t, C_t
