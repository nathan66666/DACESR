import torch

# 加载模型权重文件
model_path = "/lxy/DASR/experiments/train_vimsr_ram_test/models/net_g_400.pth"
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))  # 将模型加载到CPU上
print(checkpoint.keys())
# 查看模型权重参数
for param_tensor in checkpoint['params'].keys():
    print(f"Parameter name: {param_tensor} | Shape: {checkpoint['params'][param_tensor].shape}")

# for param_tensor in checkpoint['params_ema'].keys():
#     print(f"Parameter name: {param_tensor} | Shape: {checkpoint['params'][param_tensor].shape}")
