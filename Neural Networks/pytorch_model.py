import torch
import torch.nn as nn


class CnnModel(nn.Module):
    def __init__(self):
        super(CnnModel, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding="valid",
                bias=False,
            ),
            nn.SiLU(),
            nn.BatchNorm2d(16),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding="valid",
                bias=False,
            ),
            nn.SiLU(),
            nn.BatchNorm2d(32),
            nn.Flatten(),
        )

        self.fc1 = nn.Sequential(nn.Linear(32 * 24 * 24, 64, bias=True), nn.SiLU(),)
        self.fc2 = nn.Sequential(nn.Linear(64, 32, bias=True), nn.SiLU(),)
        self.fc3 = nn.Sequential(nn.Linear(32, 10, bias=True), nn.Softmax(),)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class CNN(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CnnModel().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

    def fit(self, x, y, test_x, test_y, batch_size=4, epochs=10):
        """
        First, prepare the data so PyTorch can use it
        Second, iterate over the training set
            Third, clear the old gradients
            Fourth, compute the forward pass and backprop the error
        Fifth, iterate over the test set to get the accuracy
        Sixth, print the loss
        """
        train_loader, test_loader = self.prepare_data(x, y, test_x, test_y, batch_size)

        for epoch in range(epochs):
            train_loss, test_loss = 0.0, 0.0

            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # zero out the parameter gradients
                # self.optimizer.zero_grad() is slow, avoid it
                for param in self.model.parameters():
                    param.grad = None

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()

            print(
                f"Iteration: {epoch}\tTrainLoss: {train_loss:.4f}\tTestLoss: {test_loss:.4f}"
            )

    def prepare_data(self, x, y, test_x, test_y, batch_size):

        # Move our data to tensors
        tensor_x = torch.Tensor(x)
        tensor_y = torch.Tensor(y).type(torch.LongTensor)
        test_tensor_x = torch.Tensor(test_x)
        test_tensor_y = torch.Tensor(test_y).type(torch.LongTensor)

        # Move our data to CUDA (if needed)
        tensor_x = tensor_x.to(self.device)
        tensor_y = tensor_y.to(self.device)
        test_tensor_x = test_tensor_x.to(self.device)
        test_tensor_y = test_tensor_y.to(self.device)

        # Change the input size from (_, 28, 28) to (_, 1, 28, 28)
        tensor_x = tensor_x.view(tensor_x.size(0), 1, 28, 28)
        test_tensor_x = test_tensor_x.view(test_tensor_x.size(0), 1, 28, 28)

        # Move the data to dataloaders
        train = torch.utils.data.TensorDataset(tensor_x, tensor_y)
        test = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)

        train_loader = torch.utils.data.DataLoader(
            train, batch_size=batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test, batch_size=batch_size, shuffle=True
        )

        return train_loader, test_loader
