import torch
import torch.nn as nn

# linear regression
# f = w * x
# here: f = 2 * x

X = torch.tensor([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8], [10], [12], [14], [16]], dtype=torch.float32)

n_samples, n_features = X.shape
print(f'n_sample = {n_samples}, n_features = {n_features}')

# sample test
X_test = torch.tensor([10], dtype=torch.float32)

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)
    
input_size, output_size = n_features, n_features
model = LinearRegression(input_size, output_size)

print(f'Prediction before training: f(10) = {model(X_test).item():.3f}')

learning_rate = 0.01
n_epochs = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_epochs):
    # prediction = forward pass
    y_pred = model(X)

    # loss
    l = loss(Y, y_pred)

    # gradients = backward pass
    l.backward()

    optimizer.step()

    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        w, b = model.parameters() #unpack parameters
        print(f'w: {w.shape}')
        print(f'epoch {epoch + 1}: w = {w[0][0].item():.3f}, loss = {l.item():.8f}')

print(f'Prediction after training: f(10) = {model(X_test).item():.3f}')
