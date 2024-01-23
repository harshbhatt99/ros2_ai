#!/usr/bin/env python3

# Import libraries
import rclpy
from std_msgs.msg import Float64
import torch
import torch.nn as nn

# Define the autoencoder class
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Define the encoder and decoder layers with one linear layer each having one neuron
        self.encoder = nn.Linear(1, 1)
        self.decoder = nn.Linear(1, 1) 
        # Initialize the encoder and decoder weights using Xavier uniform initialization
        """
        Xavier initialization is a technique used to initialize the weights of a neural network layer in a way 
        that helps with the convergence of the training process. It takes into account the number of input and 
        output neurons to scale the initial weights appropriately.
        The underscore at the end (xavier_uniform_) indicates that the operation is performed in-place, 
        modifying the weights of the self.encoder layer directly.
        """
        torch.nn.init.xavier_uniform_(self.encoder.weight)  
        torch.nn.init.xavier_uniform_(self.decoder.weight)  
        # Enable anomaly detection in autograd
        # This line is optional as it is to traceback to the info that causes the NaN values in the reconstruction
        torch.autograd.set_detect_anomaly(True)

    # The forward function defines the forward pass of the autoencoder
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AutoencoderNode:
    def __init__(self):
        self.node = rclpy.create_node('autoencoder_node')
        # Define the autoencoder, loss function, and optimizer
        self.autoencoder = Autoencoder()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.autoencoder.parameters(), lr=0.01)
        # Subscribe to the /number_input topic
        self.number_sub = self.node.create_subscription(
            Float64,
            '/number_input',
            self.number_callback,
            10
        )

    def number_callback(self, msg):
        # Define the input number and reconstruct the number using the autoencoder
        input_number = torch.tensor([[msg.data]], dtype=torch.float32)
        reconstructed_number = self.train_autoencoder(input_number)
        print(f"Received: {input_number}, Reconstructed: {reconstructed_number}")
        #print(f"Received: {input_number.item()}, Reconstructed: {reconstructed_number.item()}")
        #self.node.get_logger().info(f"Received: {input_number.item()}, Reconstructed: {reconstructed_number.item()}")

    def train_autoencoder(self, input_data):
        # Training the autoencoder (you can customize this part based on your needs)
        self.optimizer.zero_grad()
        output_data = self.autoencoder(input_data)
        loss = self.criterion(output_data, input_data)
        # The following line is for backward propagation
        loss.backward()
        # The following line is for gradient clipping
        """
        It clips gradients to prevent the exploding gradient problem, which can occur during the training of 
        deep neural networks. When gradients become too large, they can lead to numerical instability and make 
        the training process difficult.
        """
        torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), max_norm=0.5)
        print(f"Loss: {loss.item()}")
        # Moving to the next step of the training process
        self.optimizer.step()
        # Return the reconstructed number
        return output_data.item()

def main(args=None):
    rclpy.init(args=args)
    node = AutoencoderNode()
    rclpy.spin(node.node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
