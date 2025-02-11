import torch

# Load the checkpoint
#checkpoint_path = '/home/ubuntu/workspace/mambavision_1/mambavision/model_weights/mambavision_tiny_1k.pth.tar'
checkpoint_path = '/home/ubuntu/workspace/mambavision_1/mambavision/model_weights/checkpoint-51.pth.tar'
#import ipdb; ipdb.set_trace()
# checkpoint_path = '/home/ubuntu/workspace/mambavision_1/mambavision/model_weights/checkpoint-3.pth.tar'
checkpoint = torch.load(checkpoint_path,  map_location=torch.device('cpu')) # dictionary

# print(checkpoint)
# Extract the state dictionary
state_dict = checkpoint['state_dict']

# Count the total number of parameters
total_params = sum(param.numel() for param in state_dict.values())

print(f"Total number of parameters: {total_params}")
