import torch
import torch.nn as nn
from settings import ARM_CONFIG

low = 250.0
high = 600.0

X1 = torch.empty(100, 2).uniform_(low, high)
 

Z = torch.full((100,), -20).unsqueeze(1) 

X= torch.cat([X1, Z], dim=1)


low,high = ARM_CONFIG['joint_limits']['shoulder_rotate']
Y = torch.randint(low, high, (100, 1)).unsqueeze(1)


low,high = ARM_CONFIG['joint_limits']['shoulder_move']
y1 = torch.randint(low, high, (100, 1)).unsqueeze(1)
Y = torch.cat([Y, y1], dim=1)
low,high = ARM_CONFIG['joint_limits']['arm_rotate']
y1 = torch.randint(low, high, (100, 1)).unsqueeze(1)
Y = torch.cat([Y, y1], dim=1)
low,high = ARM_CONFIG['joint_limits']['elbow_move']
y1 = torch.randint(low, high, (100, 1)).unsqueeze(1)
Y = torch.cat([Y, y1], dim=1)
low,high = ARM_CONFIG['joint_limits']['elbow_rotate']
y1 = torch.randint(low, high, (100, 1)).unsqueeze(1)
Y = torch.cat([Y, y1], dim=1)
low,high = ARM_CONFIG['joint_limits']['wrist_move']
y1 = torch.randint(low, high, (100, 1)).unsqueeze(1)
Y = torch.cat([Y, y1], dim=1)




low = 250.0
high = 600.0

X1 = torch.empty(100, 2).uniform_(low, high)
 

Z = torch.full((100,), -20).unsqueeze(1) 

Xtest= torch.cat([X1, Z], dim=1)


low,high = ARM_CONFIG['joint_limits']['shoulder_rotate']
Ytest = torch.randint(low, high, (100, 1)).unsqueeze(1)


low,high = ARM_CONFIG['joint_limits']['shoulder_move']
y1 = torch.randint(low, high, (100, 1)).unsqueeze(1)
Ytest = torch.cat([Y, y1], dim=1)
low,high = ARM_CONFIG['joint_limits']['arm_rotate']
y1 = torch.randint(low, high, (100, 1)).unsqueeze(1)
Ytest = torch.cat([Y, y1], dim=1)
low,high = ARM_CONFIG['joint_limits']['elbow_move']
y1 = torch.randint(low, high, (100, 1)).unsqueeze(1)
Ytest = torch.cat([Y, y1], dim=1)
low,high = ARM_CONFIG['joint_limits']['elbow_rotate']
y1 = torch.randint(low, high, (100, 1)).unsqueeze(1)
Ytest = torch.cat([Y, y1], dim=1)
low,high = ARM_CONFIG['joint_limits']['wrist_move']
y1 = torch.randint(low, high, (100, 1)).unsqueeze(1)
Ytest = torch.cat([Y, y1], dim=1)



