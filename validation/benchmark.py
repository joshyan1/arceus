import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StandardMNIST(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Create the same architecture as distributed version
        for i in range(len(layer_sizes)-1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes)-2:
                self.layers.append(nn.ReLU())
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        for layer in self.layers:
            x = layer(x)
        return x

class Benchmark:
    def __init__(self, layer_sizes=[784, 128, 64, 10]):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.layer_sizes = layer_sizes
        self.model = StandardMNIST(layer_sizes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)
        
        # Match distributed training settings
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        self.timing_stats = {
            'forward_time': [],
            'backward_time': [],
            'update_time': [],
            'data_prep_time': [],
            'total_time': []
        }
    
    def load_data(self):
        logger.info("Loading MNIST dataset...")
        train_dataset = datasets.MNIST('data', train=True, download=True, transform=self.transform)
        val_dataset = datasets.MNIST('data', train=False, transform=self.transform)
        
        self.train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=1000)
    
    def train_epoch(self, epoch, epochs):
        self.model.train()
        epoch_loss = 0
        epoch_acc = 0
        n_batches = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            batch_start = time.time()
            
            # Data preparation timing
            prep_start = time.time()
            data, target = data.to(self.device), target.to(self.device)
            self.timing_stats['data_prep_time'].append(time.time() - prep_start)
            
            self.optimizer.zero_grad()
            
            # Forward pass timing
            forward_start = time.time()
            output = self.model(data)
            self.timing_stats['forward_time'].append(time.time() - forward_start)
            
            # Loss computation and backward pass timing
            backward_start = time.time()
            loss = self.criterion(output, target)
            loss.backward()
            self.timing_stats['backward_time'].append(time.time() - backward_start)
            
            # Update timing
            update_start = time.time()
            self.optimizer.step()
            self.timing_stats['update_time'].append(time.time() - update_start)
            
            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            acc = pred.eq(target.view_as(pred)).float().mean().item()
            
            epoch_loss += loss.item()
            epoch_acc += acc
            n_batches += 1
            
            self.timing_stats['total_time'].append(time.time() - batch_start)
            
            if (batch_idx + 1) % 10 == 0:
                self.print_stats(epoch, epochs, batch_idx)
        
        return epoch_loss / n_batches, epoch_acc / n_batches
    
    def print_stats(self, epoch, epochs, batch_idx):
        avg_forward = sum(self.timing_stats['forward_time'][-10:]) / 10
        avg_backward = sum(self.timing_stats['backward_time'][-10:]) / 10
        avg_update = sum(self.timing_stats['update_time'][-10:]) / 10
        avg_prep = sum(self.timing_stats['data_prep_time'][-10:]) / 10
        avg_total = sum(self.timing_stats['total_time'][-10:]) / 10
        
        logger.info(
            f"\nEpoch {epoch+1}/{epochs} "
            f"[Batch {batch_idx+1}/{len(self.train_loader)}]"
        )
        logger.info(f"Forward pass: {avg_forward:.4f}s")
        logger.info(f"Backward pass: {avg_backward:.4f}s")
        logger.info(f"Parameter update: {avg_update:.4f}s")
        logger.info(f"Data preparation: {avg_prep:.4f}s")
        logger.info(f"Total batch time: {avg_total:.4f}s")
        logger.info("-" * 50)
    
    def validate(self):
        self.model.eval()
        val_loss = 0
        val_acc = 0
        n_batches = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                val_acc += pred.eq(target.view_as(pred)).float().mean().item()
                n_batches += 1
        
        return val_loss / n_batches, val_acc / n_batches
    
    def save_stats(self):
        stats_dir = Path("validation/metrics/")
        stats_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        stats_file = stats_dir / f"benchmark_stats_{timestamp}.json"
        
        with open(stats_file, 'w') as f:
            json.dump(self.timing_stats, f, indent=2)
        logger.info(f"Saved benchmark stats to {stats_file}")
    
    def run_benchmark(self, epochs=10):
        logger.info(f"Starting benchmark training on {self.device}")
        logger.info(f"Model architecture: {self.layer_sizes}")
        
        self.load_data()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            train_loss, train_acc = self.train_epoch(epoch, epochs)
            val_loss, val_acc = self.validate()
            epoch_time = time.time() - epoch_start
            
            logger.info(
                f"\nEpoch {epoch+1} Summary:"
                f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
                f"\nVal Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
                f"\nEpoch Time: {epoch_time:.2f}s"
                f"\n{'-' * 50}"
            )
        
        self.save_stats()

def main():
    benchmark = Benchmark()
    benchmark.run_benchmark()

if __name__ == "__main__":
    main() 