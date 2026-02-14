import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from collections import OrderedDict
import flwr as fl
from src.dataset import load_data, get_client_data, get_test_loader
from flwr.client import ClientApp
from flwr.common import Context

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(13, 32)    #initially 64
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))   
        x = F.relu(self.fc2(x))  
        x = torch.sigmoid(self.fc3(x))
        return x

def train(net, trainloader, epochs, lr):
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()     
            loss = criterion(net(images), labels) 
            loss.backward()         
            optimizer.step()     

def test(net, testloader):
    """Validate the model on local test data."""
    criterion = nn.BCELoss()
    correct, total, loss = 0, 0, 0.0
    
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return loss / len(testloader), correct / total

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, testloader):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        lr = config.get("lr", 0.01)
        train(self.net, self.trainloader, epochs=5, lr=lr)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {} 

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}

def client_fn(context: Context):
    """Create a Flower client."""
    X_train, X_test, y_train, y_test = load_data()
    
    # Get client_id from context instead of command line
    client_id = context.node_config["partition-id"]
    num_clients = context.node_config["num-partitions"]
    
    print(f"Starting Client {client_id}...")
    
    trainloader = get_client_data(client_id, num_clients, X_train, y_train)
    testloader = get_test_loader(X_test, y_test)
    
    net = Net()
    return FlowerClient(net, trainloader, testloader).to_client()

app = ClientApp(client_fn=client_fn)