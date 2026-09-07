import torch
import torch.nn as nn
import torch.optim as optim

# 1. Generate some dummy data (X inputs, Y targets)
# Let's say we have 100 samples, 5 input features, and 1 binary output (0 or 1)
X = torch.randn(100, 5)
Y = torch.randint(0, 2, (100, 1)).float()

# 2. Define the Neural Network structure
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Hidden layer: takes 5 inputs, outputs 8 features
        self.hidden = nn.Linear(5, 8)  
        # Output layer: takes 8 features, outputs 1 probability value
        self.output = nn.Linear(8, 1)  
        # Activation function
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

# Initialize the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.BCELoss() # Binary Cross Entropy Loss for binary classification
optimizer = optim.SGD(model.parameters(), lr=0.1) # Stochastic Gradient Descent

# 3. The Training Loop
epochs = 20

for epoch in range(epochs):
    # Reset gradients to zero
    optimizer.zero_grad()
    
    # Forward pass: compute predicted outputs by passing X to the model
    predictions = model(X)
    
    # Calculate the loss
    loss = criterion(predictions, Y)
    
    # Backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()
    
    # Update weights based on the current gradients
    optimizer.step()
    
    # Print progress every 5 epochs
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
