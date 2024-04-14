import torch
import torch.nn as nn


class LabelNet(nn.Module):
    def __init__(self):
        super(LabelNet, self).__init__()

    def forward(self, x):
        # Compute the mean of all elements in the input tensor
        mean_value = torch.mean(x)
        return mean_value


# Example usage:
if __name__ == "__main__":
    # Create a dummy input tensor of size [1, 3, 512, 512]
    input_tensor = torch.randn(1, 3, 512, 512)

    # Instantiate the model
    model = LabelNet()

    # Forward pass: compute the output
    output = model(input_tensor)

    print("Output mean value:", output.item())
