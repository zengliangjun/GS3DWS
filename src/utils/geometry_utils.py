import torch
import torch.nn.functional as F
import numpy as np

def _gradient(img, normalize_gradient=True):
    """ Calculate the gradient on the feature space using Sobel operator
    :param the input image
    -----------
    :return the gradient of the image in x, y direction
    """
    H, W, C = img.shape
    # to filter the image equally in each channel
    wx = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3).type_as(img)
    wy = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3).type_as(img)

    img_permuted = img.permute(2, 0, 1).view(-1, 1, H, W)  # [c, 1, h, w]
    img_pad = F.pad(img_permuted, (1, 1, 1, 1), mode='replicate')
    img_dx = F.conv2d(img_pad, wx, stride=1, padding=0).squeeze().permute(1, 2, 0)  # [h, w, c]
    img_dy = F.conv2d(img_pad, wy, stride=1, padding=0).squeeze().permute(1, 2, 0)  # [h, w, c]

    if normalize_gradient:
        mag = torch.sqrt((img_dx ** 2) + (img_dy ** 2) + 1e-8)
        img_dx = img_dx / mag
        img_dy = img_dy / mag

    return img_dx, img_dy  # [h, w, c]

def vertex(depth, K):
    assert isinstance(depth, torch.Tensor)
    H, W = depth.shape[:2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    device = depth.device

    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t().to(device)  # [h, w]
    j = j.t().to(device)  # [h, w]

    vertex = torch.stack([(i - cx) / fx, (j - cy) / fy, torch.ones_like(i)], -1).to(device) * depth  # [h, w, 3]
    return vertex

def normal(vertex_map):
    """ Calculate the normal map from a depth map
    :param the input depth image
    -----------
    :return the normal map
    """
    assert isinstance(vertex_map, torch.Tensor)
    H, W, C = vertex_map.shape
    img_dx, img_dy = _gradient(vertex_map, normalize_gradient=False)  # [h, w, 3]

    normal = torch.cross(img_dy.view(-1, 3), img_dx.view(-1, 3))
    normal = normal.view(H, W, 3)  # [h, w, 3]

    mag = torch.norm(normal, p=2, dim=-1, keepdim=True)
    normal = normal / (mag + 1e-8)

    # filter out invalid pixels
    depth = vertex_map[:, :, -1]
    # 0.5 and 5.
    invalid_mask = (depth <= depth.min()) | (depth >= depth.max())
    zero_normal = torch.zeros_like(normal)
    normal = torch.where(invalid_mask[..., None], zero_normal, normal)
    return normal

def confidence(normal_map, intrinsic):
    _device = normal_map.device
    H, W, C = normal_map.shape
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    h_grid, w_grid = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")

    proj_map = torch.ones(H, W, 3).to(_device)
    proj_map[..., 0] = (w_grid.to(_device) - cx) / fx
    proj_map[..., 1] = (h_grid.to(_device) - cy) / fy
    mag = torch.norm(proj_map, p=2, dim=-1, keepdim=True)
    proj_map = proj_map / (mag + 1e-8)
    view_normal_dist = torch.abs(F.cosine_similarity(normal_map, proj_map, dim=-1))
    return view_normal_dist[..., None]

def homogeneous(points: torch.tensor):
    temp = points[..., :1]
    ones = torch.ones_like(temp).type(torch.float32)
    return torch.cat([points, ones], dim=-1)

def transform(map: torch.tensor, transform: torch.tensor):
    assert map.shape[-1] == 1 or map.shape[-1] == 3
    H, W, C = map.shape[0], map.shape[1], map.shape[2]
    transform_expand = transform.unsqueeze(0).unsqueeze(0).expand(H, W, -1, -1)
    _map = torch.matmul(transform_expand, homogeneous(map).unsqueeze(-1)).squeeze()[..., :C]
    return _map

def compute_rot(init_vec, target_vec):
    axis = torch.cross(init_vec, target_vec)
    axis = axis / (torch.norm(axis, p=2, dim=-1, keepdim=True) + 1e-8)
    angle = torch.acos(torch.sum(init_vec * target_vec, dim=1)).unsqueeze(-1)
    from pytorch3d.transforms import axis_angle_to_quaternion
    rots = axis_angle_to_quaternion(axis * angle)
    return rots

################################################
def compare_rotmat(prev_rot: np.array, curr_rot: np.array):
    rot_diff = prev_rot.T @ curr_rot
    cos_theta = (np.trace(rot_diff) - 1) / 2
    rad_diff = np.arccos(cos_theta)
    theta_diff = np.rad2deg(rad_diff)
    return rad_diff, theta_diff


def compare_trans(prev_trans: np.array, curr_trans: np.array):
    trans_diff = prev_trans - curr_trans
    l1_diff = np.linalg.norm(trans_diff, ord=1)
    l2_diff = np.linalg.norm(trans_diff, ord=2)
    return l1_diff, l2_diff