from mann_cell import MANNCell
import torch
import torch.nn as nn
import torch.nn.functional as F

class MANN(nn.Module):
    def __init__(self, learning_rate=1e-3, input_size=20*20, memory_size=128, memory_dim=40,
                 controller_size=200, nb_reads=4, num_layers=1, nb_classes=5, nb_samples_per_class=10, 
                 batch_size=16, device='cpu', model="MANN"):
        super(MANN, self).__init__()
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.controller_size = controller_size
        self.nb_reads = nb_reads
        self.num_layers = num_layers
        self.nb_classes = nb_classes
        self.nb_samples_per_class = nb_samples_per_class
        self.batch_size = batch_size
        self.model = model
        self.device = device

        if self.model == "LSTM":
            self.cell = nn.LSTM(input_size=self.input_size + self.nb_classes, hidden_size=self.controller_size, num_layers=self.num_layers).to(self.device)
            self.hidden_dim = self.controller_size
        elif self.model == "MANN":
            self.cell = MANNCell(lstm_size=self.controller_size, memory_size=self.memory_size, memory_dim=self.memory_dim, 
                                nb_reads=self.nb_reads, input_dim=self.input_size + self.nb_classes).to(device)
            self.hidden_dim = self.controller_size + self.nb_reads * self.memory_dim

        self.output_layer = nn.Linear(self.hidden_dim, self.nb_classes)

    def forward(self, input_var, target_var):
        target_var = target_var.view(self.batch_size, -1).long()
        one_hot_target = F.one_hot(target_var, num_classes=self.nb_classes).float()
        offset_target_var = torch.cat([torch.zeros(self.batch_size, 1, self.nb_classes, device=input_var.device), one_hot_target[:, :-1]], dim=1)
        ntm_input = torch.cat([input_var, offset_target_var], dim=-1)

        prev_state = self.cell.zero_state(self.batch_size, self.device)

        outputs = []
        for t in range(ntm_input.size(1)):
            output_t, updated_state = self.cell(ntm_input[:, t, :], prev_state)
            prev_state = updated_state
            outputs.append(output_t)

        outputs = torch.stack(outputs, dim=1)
        output = self.output_layer(outputs)
        output = F.softmax(output, dim=-1)
        output = output.view(self.batch_size, self.nb_classes * self.nb_samples_per_class, self.nb_classes)

        return output
