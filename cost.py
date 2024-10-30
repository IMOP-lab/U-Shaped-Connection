### script for test model parameters and FLOPs

import torch
import time
import os
from model import get3dmodel
from thop import profile

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

network_name = 'uC_SegResNet'

model = get3dmodel(network_name, 1, 5)  # Make sure this line initializes your model correctly
print('network_name',network_name)

dummy_input = torch.randn(1 ,1, 96, 96, 96)
flops, params = profile(model, (dummy_input,))
print('params: ', params, 'flops: ', flops)
print('params: %.2f M, flops: %.2f G' % (params / 1000000.0, flops / 1000000000.0))

# Transfer the model to the appropriate device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:",device)
model.to(device)

model.eval()
# Generate a data samples
inputs = torch.randn(1, 1, 96, 96, 96).to(device)  # Adjust the size according to your model input
ass = model(inputs)
# Record start time to measure model inference speed
start_time = time.time()

# Processing samples
for i in range(100):  # Adding an extra batch dimension
    ass = model(inputs)
# Calculate total inference time
end_time = time.time()
total_time = end_time - start_time

print(f"Time required to process 1 samples:{total_time*10}ms")
print(f"Time required to process 100 samples:{total_time}s")
