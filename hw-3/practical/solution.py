import random
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.transforms.functional import to_tensor, normalize, affine
from PIL import Image
from typing import Tuple, List, NamedTuple
from tqdm import tqdm
import zipfile
from urllib import request


# Seed all random number generators
np.random.seed(197331)
torch.manual_seed(197331)
random.seed(197331)


class NetworkConfiguration(NamedTuple):
    n_channels: Tuple[int, ...] = (16, 32, 48)
    kernel_sizes: Tuple[int, ...] = (3, 3, 3)
    strides: Tuple[int, ...] = (1, 1, 1)
    dense_hiddens: Tuple[int, ...] = (256, 256)


class Trainer:
    def __init__(self,
                 network_type: str = "mlp",
                 net_config: NetworkConfiguration = NetworkConfiguration(),
                 lr: float = 0.001,
                 batch_size: int = 128,
                 activation_name: str = "relu"):
        self.train, self.test = self.load_dataset()
        self.network_type = network_type
        activation_function = self.create_activation_function(activation_name)
        input_dim = self.train[0].shape[1:]
        if network_type == "mlp":
            self.network = self.create_mlp(input_dim[0]*input_dim[1]*input_dim[2], 
                                           net_config,
                                           activation_function)
        elif network_type == "cnn":
            self.network = self.create_cnn(input_dim[0], 
                                           net_config, 
                                           activation_function)
        else:
            raise ValueError("Network type not supported")
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.lr = lr
        self.batch_size = batch_size

        self.train_logs = {'train_loss': [], 'test_loss': [],
                           'train_mae': [], 'test_mae': []}

    @staticmethod
    def load_dataset() -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        if not os.path.exists('./rotated_fashion_mnist'):
            url = 'https://drive.google.com/u/0/uc?id=1NQPmr01eIafQKeH9C9HR0lGuB5z6mhGb&export=download&confirm=t&uuid=645ff20a-d47b-49f0-ac8b-4a7347529c8e&at=AHV7M3d_Da0D7wowJlTzzZxDky5c:1669325231545'
            with request.urlopen(url) as f:
                with open('./rotated_fashion_mnist.zip', 'wb') as out:
                    out.write(f.read())
            with zipfile.ZipFile('./rotated_fashion_mnist.zip', 'r') as zip_ref:
                zip_ref.extractall()
            os.remove('./rotated_fashion_mnist.zip')
            
        datapath = './rotated_fashion_mnist'

        def get_paths_and_rots(split: str) -> List[Tuple[str, float]]:
            image_paths, rots = [], []
            files = os.listdir(os.path.join(datapath, split))
            for file in files:
                image_paths.append(os.path.join(datapath, split, file))
                rots.append(float(file.split('_')[1].split('.')[0]))
            return image_paths, rots
        
        def to_tensors(image_paths: List[str], rots: List[float]) -> Tuple[torch.Tensor, torch.Tensor]:
            images = [normalize(to_tensor(Image.open(path)), (0.5,), (0.5,)) 
                      for path in image_paths]
            images = torch.stack(images)
            labels = torch.tensor(rots).view(-1, 1)
            return images, labels

        X_train, y_train = to_tensors(*get_paths_and_rots('train'))
        X_test, y_test = to_tensors(*get_paths_and_rots('test'))
        
        # Normalize y for easier training
        mean, std = y_train.mean(), y_train.std()
        y_train = (y_train - mean) / std
        y_test = (y_test - mean) / std
        
        return (X_train, y_train), (X_test, y_test)

    @staticmethod
    def create_mlp(input_dim: int, net_config: NetworkConfiguration,
                   activation: torch.nn.Module) -> torch.nn.Module:
        """
        Create a multi-layer perceptron (MLP) network.

        :param net_config: a NetworkConfiguration named tuple. Only the field 'dense_hiddens' will be used.
        :param activation: The activation function to use.
        :return: A PyTorch model implementing the MLP.
        """
        # TODO write code here

        modules = []
        # Flatten from dimension 1 onwards (dimension 0 is the batch index)
        modules.append(torch.nn.Flatten(start_dim=1))

        for idx in range(len(net_config.dense_hiddens)):
            # Hidden layer
            if idx == 0:
                modules.append(torch.nn.Linear(input_dim, net_config.dense_hiddens[idx]))
            else:
                modules.append(torch.nn.Linear(net_config.dense_hiddens[idx-1], net_config.dense_hiddens[idx]))

            # Activation layer after every hidden layer
            modules.append(activation)

        # Final layer
        modules.append(torch.nn.Linear(net_config.dense_hiddens[-1], 1))


        layers = torch.nn.Sequential(*modules)
        return layers

    @staticmethod
    def create_cnn(in_channels: int, net_config: NetworkConfiguration,
                   activation: torch.nn.Module) -> torch.nn.Module:
        """
        Create a convolutional network.

        :param in_channels: The number of channels in the input image.
        :param net_config: a NetworkConfiguration specifying the architecture of the CNN.
        :param activation: The activation function to use.
        :return: A PyTorch model implementing the CNN.
        """
        # TODO write code here
        modules = []

        for idx in range(len(net_config.n_channels)):
            # Conv layer
            if idx == 0:
                modules.append(torch.nn.Conv2d(in_channels, net_config.n_channels[idx], net_config.kernel_sizes[idx], net_config.strides[idx]))
            else:
                modules.append(torch.nn.Conv2d(net_config.n_channels[idx-1], net_config.n_channels[idx], net_config.kernel_sizes[idx], net_config.strides[idx]))

            # Activation layer
            modules.append(activation)

            # Max Pooling layer, but last convolution layer should not have max pooling after it
            if idx < len(net_config.n_channels) - 1:
                modules.append(torch.nn.MaxPool2d(kernel_size=2))
            else:
                # AdaptiveMaxPool forces output to be batch_size x channels x 4 x 4
                modules.append(torch.nn.AdaptiveMaxPool2d((4, 4)))
                
        for idx in range(len(net_config.dense_hiddens)):
            # Fully-connected layer
            if idx == 0:
                # Flatten input before fully-connected layer
                modules.append(torch.nn.Flatten())
                num_in_features = net_config.n_channels[-1] * 4 * 4
                modules.append(torch.nn.Linear(in_features=num_in_features, out_features=net_config.dense_hiddens[idx]))
            else:
                modules.append(torch.nn.Linear(in_features=net_config.dense_hiddens[idx-1], out_features=net_config.dense_hiddens[idx]))
                
            # Activation
            modules.append(activation)

        # Final fully-connected layer
        modules.append(torch.nn.Linear(net_config.dense_hiddens[-1], 1))

        layers = torch.nn.Sequential(*modules)
        return layers

    @staticmethod
    def create_activation_function(activation_str: str) -> torch.nn.Module:
        if activation_str == 'relu':
            return Relu()
        elif activation_str == 'tanh':
            return Tanh()
        elif activation_str == 'sigmoid':
            return Sigmoid()
        else:
            raise ValueError(f"Unknown activation_str given: {activation_str}")

    def compute_loss_and_mae(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO WRITE CODE HERE
        outputs = self.network(X)
        mse = torch.mean(torch.square(outputs - y))
        mae = torch.mean(torch.abs(outputs - y))

        return mse, mae

    def training_step(self, X_batch: torch.Tensor, y_batch: torch.Tensor):
        # TODO WRITE CODE HERE
        # Zero the parameter gradients
        self.optimizer.zero_grad()

        # Compute loss and backpropagate gradients
        mse_loss, mae = self.compute_loss_and_mae(X_batch, y_batch)
        mse_loss.backward()
        self.optimizer.step()

    def log_metrics(self, X_train: torch.Tensor, y_train: torch.Tensor,
                    X_test: torch.Tensor, y_test: torch.Tensor) -> None:
        self.network.eval()
        with torch.inference_mode():
            train_loss, train_mae = self.compute_loss_and_mae(X_train, y_train)
            test_loss, test_mae = self.compute_loss_and_mae(X_test, y_test)
        self.train_logs['train_mae'].append(train_mae.item())
        self.train_logs['test_mae'].append(test_mae.item())
        self.train_logs['train_loss'].append(train_loss.item())
        self.train_logs['test_loss'].append(test_loss.item())

    def train_loop(self, n_epochs: int):
        # Prepare train and validation data
        X_train, y_train = self.train
        X_test, y_test = self.test

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

        self.log_metrics(X_train[:2000], y_train[:2000], X_test, y_test)
        for epoch in tqdm(range(n_epochs)):
            for batch in range(n_batches):
                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                self.training_step(minibatchX, minibatchY)
            self.log_metrics(X_train[:2000], y_train[:2000], X_test, y_test)
        return self.train_logs

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO WRITE CODE HERE
        self.network.eval()
        with torch.inference_mode():
            mse_loss, mae = self.compute_loss_and_mae(X, y)

        return mse_loss, mae

    def test_equivariance(self):
        from functools import partial

        test_im = (self.train[0][0] + 1) / 2
        conv = torch.nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1, stride=1, padding=0)
        fullconv_model = lambda x: torch.relu(conv((torch.relu(conv((x))))))
        model = fullconv_model

        shift_amount = 5
        shift = partial(affine, angle=0, translate=(shift_amount, shift_amount), scale=1, shear=0)
        rotation = partial(affine, angle=90, translate=(0, 0), scale=1, shear=0)

        # TODO CODE HERE
        
        # No need for Autograd functionality, turn it off to make cloning tensors easier
        with torch.no_grad():
            # Plot image as-is
            img = test_im.clone().detach()

            # Plot model output 
            model_output = model(test_im.clone())

            # Unshifted input, shifted model output
            shifted_model_output = shift(model_output.clone())

            # Shifted input, unshifted model output
            shifted_input = shift(test_im.clone())
            shifted_input_output = model(shifted_input)

            shifted_difference = torch.abs(shifted_model_output - shifted_input_output)

            # Unrotated input, rotated model output
            rotated_model_output = rotation(model_output.clone())

            # Rotated input, unrotated model output
            rotated_input = rotation(test_im.clone())
            rotated_input_output = model(rotated_input)

            rotated_difference = torch.abs(rotated_model_output - rotated_input_output)

        return img.detach().numpy(), model_output.detach().numpy(), shifted_difference.detach().numpy(), rotated_difference.detach().numpy()


class Relu(torch.nn.Module):
    """Applies the rectified linear unit function element-wise:

    Applies the Rectified linear unit function element-wise:

    Relu(x) = max(0, x)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        output = x
        output[output < 0] = 0
        return output

class Sigmoid(torch.nn.Module):
    """Applies the sigmoid function element-wise:

    Sigmoid(x) = 1 / (1+exp(-x))
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        output = 1 / (1+torch.exp(-x))
        return output

class Tanh(torch.nn.Module):
    """Applies the Tanh function element-wise:

    Tanh(x) = (exp(x) - exp(-x)) / (exp(x)+exp(-x))
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        output = (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
        return output