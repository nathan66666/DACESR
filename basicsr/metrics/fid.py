import numpy as np
import torch
import torch.nn as nn
from scipy import linalg
from tqdm import tqdm

from basicsr.archs.inception import InceptionV3
from basicsr.utils.registry import METRIC_REGISTRY

def load_patched_inception_v3(device='cuda', resize_input=True, normalize_input=False):
    # we may not resize the input, but in [rosinality/stylegan2-pytorch] it
    # does resize the input.
    inception = InceptionV3([3], resize_input=resize_input, normalize_input=normalize_input)
    inception = nn.DataParallel(inception).eval().to(device)
    return inception


@torch.no_grad()
def extract_inception_features(data_generator, inception, len_generator=None, device='cuda'):
    """Extract inception features.

    Args:
        data_generator (generator): A data generator.
        inception (nn.Module): Inception model.
        len_generator (int): Length of the data_generator to show the
            progressbar. Default: None.
        device (str): Device. Default: cuda.

    Returns:
        Tensor: Extracted features.
    """
    if len_generator is not None:
        pbar = tqdm(total=len_generator, unit='batch', desc='Extract')
    else:
        pbar = None
    features = []

    for data in data_generator:
        if pbar:
            pbar.update(1)
        data = data.to(device)
        feature = inception(data)[0].view(data.shape[0], -1)
        features.append(feature.to('cpu'))
    if pbar:
        pbar.close()
    features = torch.cat(features, 0)
    return features

@METRIC_REGISTRY.register()
def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.

    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
        d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.

    Args:
        mu1 (np.array): The sample mean over activations.
        sigma1 (np.array): The covariance matrix over activations for
            generated samples.
        mu2 (np.array): The sample mean over activations, precalculated on an
               representative data set.
        sigma2 (np.array): The covariance matrix over activations,
            precalculated on an representative data set.

    Returns:
        float: The Frechet Distance.
    """
    assert mu1.shape == mu2.shape, 'Two mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, ('Two covariances have different dimensions')

    cov_sqrt, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    # Product might be almost singular
    if not np.isfinite(cov_sqrt).all():
        print('Product of cov matrices is singular. Adding {eps} to diagonal ' 'of cov estimates')
        offset = np.eye(sigma1.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))
            raise ValueError(f'Imaginary component {m}')
        cov_sqrt = cov_sqrt.real

    mean_diff = mu1 - mu2
    mean_norm = mean_diff @ mean_diff
    trace = np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(cov_sqrt)
    fid = mean_norm + trace

    return fid

# import numpy as np
# import torch
# import torch.nn as nn
# from torchvision import transforms
# from PIL import Image
# from scipy import linalg
# from basicsr.archs.inception import InceptionV3

# def load_patched_inception_v3(device='cuda', resize_input=True, normalize_input=False):
#     inception = InceptionV3([3], resize_input=resize_input, normalize_input=normalize_input)
#     inception = nn.DataParallel(inception).eval().to(device)
#     return inception

# @torch.no_grad()
# def extract_single_inception_feature(img, inception, device='cuda'):
#     img = img.to(device)
#     feature = inception(img.unsqueeze(0))[0].view(1, -1)  # Unsqueeze to add batch dimension
#     return feature.cpu().numpy()

# def calculate_single_image_stats(image, inception, device='cuda'):
#     feature = extract_single_inception_feature(image, inception, device=device)
#     mu = np.mean(feature, axis=0)  # Mean is just the feature itself for a single image
#     sigma = np.cov(feature, rowvar=False)  # Covariance is zero for a single image
#     return mu, sigma

# def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
#     assert mu1.shape == mu2.shape, 'Two mean vectors have different lengths'
#     assert sigma1.shape == sigma2.shape, 'Two covariances have different dimensions'
    
#     cov_sqrt, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    
#     if not np.isfinite(cov_sqrt).all():
#         offset = np.eye(sigma1.shape[0]) * eps
#         cov_sqrt = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    
#     if np.iscomplexobj(cov_sqrt):
#         cov_sqrt = cov_sqrt.real
    
#     mean_diff = mu1 - mu2
#     mean_norm = mean_diff @ mean_diff
#     trace = np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(cov_sqrt)
    
#     fid = mean_norm + trace
#     return fid

# # 图像预处理
# transform = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])

# # 加载两张图像
# img1 = Image.open('/lxy/DASR/datasets/DIV2K_val_realsr/gt/0000101.png').convert('RGB')
# img2 = Image.open('/lxy/DASR/results/test_DVMSR/visualization/DIV2K_val_realsr/0000101_test_DVMSR.png').convert('RGB')

# # 转换为 Tensor
# img1_tensor = transform(img1)
# img2_tensor = transform(img2)

# # 加载 InceptionV3 模型
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# inception = load_patched_inception_v3(device=device)

# # 计算特征均值和协方差
# mu1, sigma1 = calculate_single_image_stats(img1_tensor, inception, device)
# mu2, sigma2 = calculate_single_image_stats(img2_tensor, inception, device)

# # 计算 FID
# fid_value = calculate_fid(mu1, sigma1, mu2, sigma2)
# print(f"FID for single image pair: {fid_value}")
