import torch
import argparse
from utility import create_dataloader
from model import new_model, train_model, test_model, save_checkpoint

# Define command arguments
parser = argparse.ArgumentParser()
parser.add_argument('data_directory', type = str)
parser.add_argument('--save_dir', type = str, default='checkpoint72.pth')
parser.add_argument('--arch', type = str, default='densenet201')
parser.add_argument('--learning_rate', type = float, default=0.001)
parser.add_argument('--hidden_units', type = int, default=640)
parser.add_argument('--epochs', type = int, default=8)
parser.add_argument('--gpu', action='store_true')
args = parser.parse_args()

# Get command arguments
data_path = args.data_directory
save_path = args.save_dir
architecture = args.arch
hidden_units = args.hidden_units
learning_rate = args.learning_rate
epochs = args.epochs
gpu = args.gpu
droprate = 0.3
batch_size = 32

# Create new network
model, optimizer, criterion = new_model(architecture, hidden_units, droprate, learning_rate)
# Create dataloaders
dataset, dataloader = create_dataloader(data_path, batch_size)
# Train new network
train_model(model, optimizer, criterion, gpu, epochs, dataloader[0], dataloader[1])
# Test trained network
test_model(model, criterion, dataloader[2], gpu)
# Save network checkpoint
save_checkpoint(model, architecture, optimizer, hidden_units, droprate, learning_rate, epochs, dataset[0], save_path)