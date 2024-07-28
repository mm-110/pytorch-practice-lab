import torch

# prediction function
def forward(x):
    return w * x

# loss function: mean squared error
def loss(y, y_pred):
    return ((y_pred - y) ** 2).mean()

# linear regression
# f = w * x + b
# here: f = 2 * x

X = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8, 10, 12, 14, 16], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

X_test = 5.0

print(f'Prediction before training: f(5) = {forward(X_test).item():.3f}')

# training
learning_rate = 0.01
n_epochs = 100

for epoch in range(n_epochs):

    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)
    
    # calculate gradients = backward pass
    l.backward()

    with torch.no_grad():
        w -= learning_rate * w.grad

    # zero gradients
    w.grad.zero_()

    if (epoch + 1) % 10 == 0:
        print(f'epoch {epoch + 1}: w = {w.item():.3f}, loss = {l.item():.3f}')

print(f'Prediction after training: f({X_test}) = {forward(X_test).item():.3f}')




