import torch
import torch.nn as nn
from settings import ARM_CONFIG

# X = torch.tensor([[388, 215], [516, 173], [559, 146], [582, 113]], dtype=torch.float32)
# Y = torch.tensor([[30, 63], [40, 60], [45, 58], [50, 57]], dtype=torch.float32)

# X = torch.tensor([[250, 250], [300, 230], [320, 220], [325, 205],[330,180]], dtype=torch.float32)
# Y = torch.tensor([[30, 63],[35, 61] ,[40, 60], [45, 58], [50, 57]], dtype=torch.float32)

X = torch.tensor([[264, 132], [400, 100], [450, 50], [510, 30],[566,0]], dtype=torch.float32)
Y = torch.tensor([[30, 63],[35, 61] ,[40, 60], [45, 58], [50, 57]], dtype=torch.float32)
