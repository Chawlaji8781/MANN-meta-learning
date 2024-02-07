import os
import yaml
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import MANN
from utils.generator import OmniglotGenerator

# Load configuration from YAML file
with open('MANN/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Configuration variables
batch_size = config['batch_size']
nb_classes = config['nb_classes']
nb_samples_per_class = config['nb_samples_per_class']
input_height, input_width = config['input_height'], config['input_width']
input_size = input_height * input_width
nb_reads = config['nb_reads']
controller_size = config['controller_size']
memory_size = config['memory_size']
memory_dim = config['memory_dim']
num_layers = config['num_layers']
learning_rate = config['learning_rate']
iterations = config['iterations']
mode = config['mode']

data_generator = OmniglotGenerator(data_folder=config['data_folder'], 
                                   nb_classes=config['nb_classes'], 
                                   nb_samples_per_class=config['nb_samples_per_class'], 
                                   img_size=(config['input_height'], config['input_width']))

# Check if CUDA is available and set the device to GPU if it is, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Model, optimizer
model = MANN(learning_rate, input_size, memory_size, memory_dim, 
             controller_size, nb_reads, num_layers, 
             nb_classes, nb_samples_per_class, batch_size, device).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.NLLLoss()

# Save and load functions
def save_model(model, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(model.state_dict(), filename)

def load_model(model, filename):
    model.load_state_dict(torch.load(filename, map_location=device))

# Training function
def train(model, data_generator, optimizer, num_iterations):
    model.train()
    for ep in range(num_iterations):
        images, labels = data_generator.sample(batch_type="train", batch_size=config['batch_size'])
        images = torch.tensor(images).to(device)
        labels = torch.tensor(labels, dtype=torch.long).view(-1).to(device)
        
        optimizer.zero_grad()
        output = model(images, labels).view(-1,config['nb_classes'])
        loss = criterion(torch.log(output), labels)  # Adjusted loss calculation
        loss.backward()
        optimizer.step()

        if ep % 100 == 0:
            print(f'Epoch: {ep}, Loss: {loss.item()}')

        # Save model and log
        if ep % 5000 == 0:
            save_model(model, os.path.join(config['save_dir'], f'model_epoch_{ep}.pth'))
            os.makedirs(config['log_dir'], exist_ok=True)
            with open(os.path.join(config['log_dir'], 'training_log.csv'), 'a') as log_file:
                log_file.write(f'{ep}, {loss.item()}\n')

# Testing function
def test(model, data_generator):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for _ in range(config['test_iterations']):
            images, labels = data_generator.sample(batch_type="test", batch_size=config['batch_size'])
            images = torch.tensor(images).to(device)
            labels = torch.tensor(labels, dtype=torch.long).view(-1).to(device)
            output = model(images, labels)
            output = output.view(-1, config['nb_classes'])
            _, predicted = torch.max(output, 1)
            labels = labels.view(-1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
    accuracy = total_correct / total_samples
    print(f'Test Accuracy: {accuracy * 100}%')

# Main
if __name__ == "__main__":
    if mode == "train":
        train(model, data_generator, optimizer, config['iterations'])
        save_model(model, os.path.join(config['save_dir'], 'final_model.pth'))
    elif mode == "test":
        load_model(model, os.path.join(config['save_dir'], 'final_model.pth'))
        test(model, data_generator)

