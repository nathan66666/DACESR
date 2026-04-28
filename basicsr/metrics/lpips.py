import lpips
import torch
from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.registry import METRIC_REGISTRY
import numpy as np
loss_fn_alex = None


@METRIC_REGISTRY.register()
def calculate_lpips(img1, img2, crop_border, input_order='HWC', test_y_channel=False, strict_shape=True, **kwargs):
    """Calculate LPIPS

    Ref: https://github.com/richzhang/PerceptualSimilarity

    Args:
        img1(ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: psnr result.
    """
    global loss_fn_alex
    if loss_fn_alex is None:
        loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores

        if torch.cuda.is_available():
            loss_fn_alex.cuda()
    if strict_shape:
        assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    else:
        h, w, c = img1.shape
        img2 = img2[0:h, 0:w, 0:c]
        if img1.shape != img2.shape:
            h, w, c = img2.shape
            img1 = img1[0:h, 0:w, 0:c]
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    def np2tensor(x):
        """

        Args:
            x: RGB [0 ~ 255] HWC ndarray

        Returns: RGB [-1, 1]

        """
        return torch.tensor((x * 2 / 255.0) - 0.5).permute(2, 0, 1).unsqueeze(0).float()

    # np2tensor
    img1 = np2tensor(img1)
    img2 = np2tensor(img2)

    if torch.cuda.is_available():
        img1 = img1.cuda()
        img2 = img2.cuda()

    with torch.no_grad():
        d = loss_fn_alex(img1, img2)
    return d.view(1).cpu().numpy()[0]



# import cv2
# import math
# import numpy as np
# import lpips
# import torch
# from basicsr.utils.registry import METRIC_REGISTRY
# from basicsr.metrics.metric_util import reorder_image
# from basicsr.data.transforms import totensor
# @METRIC_REGISTRY.register()
# def calculate_lpips(img1,
#                     img2,
#                     crop_border,
#                     input_order='HWC'):
#     """Calculate LPIPS metric.

#     We use the official params estimated from the pristine dataset.
#     We use the recommended block size (96, 96) without overlaps.

#     Args:
#         img (ndarray): Input image whose quality needs to be computed.
#             The input image must be in range [0, 255] with float/int type.
#             The input_order of image can be 'HW' or 'HWC' or 'CHW'. (BGR order)
#             If the input order is 'HWC' or 'CHW', it will be converted to gray
#             or Y (of YCbCr) image according to the ``convert_to`` argument.
#         crop_border (int): Cropped pixels in each edge of an image. These
#             pixels are not involved in the metric calculation.
#         input_order (str): Whether the input order is 'HW', 'HWC' or 'CHW'.
#             Default: 'HWC'.

#     Returns:
#         float: LPIPS result.
#     """

#     assert img1.shape == img2.shape, (
#         f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
#     if input_order not in ['HWC', 'CHW']:
#         raise ValueError(
#             f'Wrong input_order {input_order}. Supported input_orders are '
#             '"HWC" and "CHW"')
#     img1 = reorder_image(img1, input_order=input_order)
#     img2 = reorder_image(img2, input_order=input_order)
#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)


#     if crop_border != 0:
#         img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
#         img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

#     img1 = img1.astype(np.float32)
#     img2 = img2.astype(np.float32)

#     img1, img2 = totensor([img1, img2], bgr2rgb=False, float32=True)

#     img1 = img1.unsqueeze(0)
#     img2 = img2.unsqueeze(0)

#     # image should be RGB, IMPORTANT: normalized to [-1,1]
#     img1 = (img1 / 255. - 0.5) * 2
#     img2 = (img2 / 255. - 0.5) * 2

#     loss_fn_alex = lpips.LPIPS(net='alex', verbose=False) # best forward scores

#     metric = loss_fn_alex(img1, img2).squeeze(0).float().detach().cpu().numpy()
#     return metric.mean()