import torch
import torch.nn as nn
import torch.nn.functional as F

class MANNCell(nn.Module):
    # Input dim needs to be added in pytorch for lstm in contrast to tensorflow which dynamically adapts to the first input size
    def __init__(self, lstm_size, memory_size, memory_dim, nb_reads, input_dim=50, gamma=0.95):
        super(MANNCell, self).__init__()
        self.lstm_size = lstm_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.nb_reads = nb_reads
        self.gamma = gamma
        self.controller_input = input_dim + nb_reads * memory_dim
        self.controller = nn.LSTMCell(self.controller_input, self.lstm_size)
        self.step = nn.Parameter(torch.tensor(0), requires_grad=False)

        self.parameter_dim_per_head = self.memory_dim * 2 + 1
        self.parameter_total_dim = self.parameter_dim_per_head * self.nb_reads

        self.linear = nn.Linear(self.lstm_size, self.parameter_total_dim)

    def forward(self, input, prev_state):
      # Unpack the previous state
      M_prev, r_prev, controller_state_prev, wu_prev, wr_prev = \
          prev_state['M'], prev_state['read_vector'], prev_state['controller_state'], prev_state['wu'], prev_state['wr']

      controller_input = torch.cat([input, r_prev], dim=-1)
      controller_hidden_t, controller_cell_t = self.controller(controller_input, controller_state_prev)

      # Here is the critical change: Repack the controller state into a tuple
      controller_state_t = (controller_hidden_t, controller_cell_t)  # Repack as a tuple

      parameter = self.linear(controller_hidden_t)

      # Extract least used memory slots
      indices_prev, wlu_prev = self.least_used(wu_prev)

      # Split parameters for each read head
      k = torch.tanh(parameter[:, 0:self.nb_reads * self.memory_dim])
      a = torch.tanh(parameter[:, self.nb_reads * self.memory_dim:2 * self.nb_reads * self.memory_dim])
      sig_alpha = torch.sigmoid(parameter[:, -self.nb_reads:])

      # Addressing mechanisms
      wr_t = self.read_head_addressing(k, M_prev)
      ww_t = self.write_head_addressing(sig_alpha, wr_prev, wlu_prev)

      # Update usage weights
      wu_t = self.gamma * wu_prev + torch.sum(wr_t, dim=1) + torch.sum(ww_t, dim=1)

      # Update the memory
      M_t = M_prev * (1 - torch.nn.functional.one_hot(indices_prev[:, -1], self.memory_size).float()).unsqueeze(2)
      M_t += torch.matmul(ww_t.transpose(1, 2), a.view(-1, self.nb_reads, self.memory_dim))

      # Read from memory
      r_t = torch.matmul(wr_t, M_t).view(-1, self.nb_reads * self.memory_dim)

      # Construct the new state
      state = {
          "M": M_t,
          "read_vector": r_t,
          "controller_state": controller_state_t,
          "wu": wu_t,
          "wr": wr_t.view(-1, self.nb_reads * self.memory_size)
      }

      NTM_output = torch.cat([controller_hidden_t, r_t], dim=-1)

      # Update step counter
      self.step += 1
      return NTM_output, state

    def read_head_addressing(self, k, M_prev, eps=1e-8):
      k = k.view(k.size(0), self.nb_reads, self.memory_dim)
      inner_product = torch.matmul(k, M_prev.transpose(-2, -1))

      k_norm = torch.sqrt(torch.sum(k**2, dim=2, keepdim=True))
      M_norm = torch.sqrt(torch.sum(M_prev**2, dim=2, keepdim=True)).transpose(-2, -1)

      norm_product = k_norm * M_norm
      K = inner_product / (norm_product + eps)
      return F.softmax(K, dim=-1)

    def write_head_addressing(self, sig_alpha, wr_prev, wlu_prev):
      sig_alpha = sig_alpha.unsqueeze(-1)
      wr_prev = wr_prev.view(wr_prev.size(0), self.nb_reads, self.memory_size)
      return sig_alpha * wr_prev + (1. - sig_alpha) * wlu_prev.unsqueeze(1).float()

    def least_used(self, w_u):
      _, indices = torch.topk(w_u, k=self.memory_size, largest=True, sorted=True)
      wlu = torch.sum(F.one_hot(indices[:, -self.nb_reads:], self.memory_size), dim=1)
      return indices, wlu

    def zero_state(self, batch_size, device='cpu'):
      M_0 = torch.ones(batch_size, self.memory_size, self.memory_dim, device=device) * 1e-6
      r_0 = torch.zeros(batch_size, self.nb_reads * self.memory_dim, device=device)

      # For LSTMCell, the initial states are typically tensors of zeros
      controller_hidden_state_0 = torch.zeros(batch_size, self.lstm_size, device=device)
      controller_cell_state_0 = torch.zeros(batch_size, self.lstm_size, device=device)

      wu_0 = self.variable_one_hot((batch_size, self.memory_size), device=device)
      wr_0 = self.variable_one_hot((batch_size, self.nb_reads * self.memory_size), device=device)

      state = {
          'M': M_0,
          'read_vector': r_0,
          'controller_state': (controller_hidden_state_0, controller_cell_state_0),
          'wu': wu_0,
          'wr': wr_0
      }
      return state

    @staticmethod
    def variable_one_hot(shape, device='cpu'):
        tensor = torch.zeros(shape, device=device)
        tensor[..., 0] = 1
        return tensor

