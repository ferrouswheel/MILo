import math
import torch
from typing import List
from utils.general_utils import build_rotation
from scene.cameras import Camera
from tqdm import tqdm


def get_gaussian_normals_from_view(view, gaussians, in_view_space=True):
    # Build rotation matrices of Gaussians
    gaussian_rots = build_rotation(gaussians._rotation)
    
    # Get the minimum scale index for each Gaussian
    gaussian_min_scale_idx = gaussians.get_scaling_with_3D_filter.min(dim=-1)[1][:, None, None].repeat(1, 3, 1)
    
    # Gather the normals as the shortest axis of the covariance matrices
    gaussian_normals = torch.gather(gaussian_rots, dim=2, index=gaussian_min_scale_idx).squeeze()
    
    # Flip the normals if they are pointing away from the camera
    gaussian_normals = gaussian_normals * torch.sign(
        (
            gaussian_normals * (view.camera_center[None] - gaussians.get_xyz)
        ).sum(dim=-1, keepdim=True)
    )
    
    if in_view_space:
        gaussian_normals = (gaussian_normals @ view.world_view_transform[:3,:3])
    
    return gaussian_normals


def transform_points_world_to_view(
    points:torch.Tensor,
    cameras:List[Camera],
    use_p3d_convention:bool=False,
):
    """Transform points from world space to view space.

    Args:
        points (torch.Tensor): Should have shape (n_cameras, N, 3).
        cameras (List[Camera]): List of Cameras. Should contain n_cameras elements.
        use_p3d_convention (bool, optional): Defaults to False.
        
    Returns:
        torch.Tensor: Has shape (n_cameras, N, 3).
    """
    world_view_transforms = torch.stack([camera.world_view_transform for camera in cameras], dim=0)  # (n_cameras, 4, 4)
    
    points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)  # (n_cameras, N, 4)
    view_points = (points_h @ world_view_transforms)[..., :3]  # (n_cameras, N, 3)
    if use_p3d_convention:
        factors = torch.tensor([[[-1, -1, 1]]], device=points.device)  # (1, 1, 3)
        view_points = factors * view_points  # (n_cameras, N, 3)
    return view_points


def transform_points_view_to_world(
    points:torch.Tensor,
    cameras:List[Camera],
    use_p3d_convention:bool=False,
):
    """Transform points from view space to world space.

    Args:
        points (torch.Tensor): Should have shape (n_cameras, N, 3).
        cameras (List[Camera]): List of Cameras. Should contain n_cameras elements.
        use_p3d_convention (bool, optional): Defaults to False.
        
    Returns:
        torch.Tensor: Has shape (n_cameras, N, 3).
    """
    view_world_transforms = torch.stack([camera.world_view_transform.inverse() for camera in cameras], dim=0)  # (n_cameras, 4, 4)
    
    if use_p3d_convention:
        factors = torch.tensor([[[-1, -1, 1]]], device=points.device)  # (1, 1, 3)
        points = factors * points  # (n_cameras, N, 3)
    points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)  # (n_cameras, N, 4)
    world_points = (points_h @ view_world_transforms)[..., :3]  # (n_cameras, N, 3)
    return world_points

        
def transform_points_to_pixel_space(
        points:torch.Tensor,
        cameras:List[Camera],
        points_are_already_in_view_space:bool=False,
        use_p3d_convention:bool=False,
        znear:float=1e-6,
        keep_float:bool=False,
    ):
        """Transform points from world space (3 coordinates) to pixel space (2 coordinates).

        Args:
            points (torch.Tensor): Should have shape (n_cameras, N, 3).
            cameras (List[Camera]): List of Cameras. Should contain n_cameras elements.
            points_are_already_in_view_space (bool, optional): Defaults to False.
            use_p3d_convention (bool, optional): Defaults to False.
            znear (float, optional): Defaults to 1e-6.

        Returns:
            torch.Tensor: Has shape (n_cameras, N, 2). 
                In pixel space, (0, 0) is the center of the left-top pixel,
                and (W-1, H-1) is the center of the right-bottom pixel.
        """
        if points_are_already_in_view_space:
            full_proj_transforms = torch.stack([camera.projection_matrix for camera in cameras])  # (n_depth, 4, 4)
            if use_p3d_convention:
                points = torch.tensor([[[-1, -1, 1]]], device=points.device) * points
        else:
            full_proj_transforms = torch.stack([camera.full_proj_transform for camera in cameras])  # (n_cameras, 4, 4)
        
        points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)  # (n_cameras, N, 4)
        proj_points = points_h @ full_proj_transforms  # (n_cameras, N, 4)
        proj_points = proj_points[..., :2] / proj_points[..., 3:4].clamp_min(znear)  # (n_cameras, N, 2)
        # proj_points is currently in a normalized space where 
        # (-1, -1) is the left-top corner of the left-top pixel,
        # and (1, 1) is the right-bottom corner of the right-bottom pixel.

        # For converting to pixel space, we need to scale and shift the normalized coordinates
        # such that (-1/2, -1/2) is the left-top corner of the left-top pixel, 
        # and (H-1/2, W-1/2) is the right-bottom corner of the right-bottom pixel.
        
        height, width = cameras[0].image_height, cameras[0].image_width
        image_size = torch.tensor([[width, height]], device=points.device)
        
        # proj_points = (1. + proj_points) * image_size / 2
        proj_points = (1. + proj_points) / 2 * image_size - 1./2.

        if keep_float:
            return proj_points        
        else:
            return torch.round(proj_points).long()


# the following functions are adopted from RaDe-GS: 
def depths_to_points(view, depthmap1, depthmap2=None):
    W, H = view.image_width, view.image_height
    fx = W / (2 * math.tan(view.FoVx / 2.))
    fy = H / (2 * math.tan(view.FoVy / 2.))
    intrins_inv = torch.tensor(
        [[1/fx, 0.,-W/(2 * fx)],
        [0., 1/fy, -H/(2 * fy),],
        [0., 0., 1.0]]
    ).float().cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W)+0.5, torch.arange(H)+0.5, indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0).reshape(3, -1).float().cuda()
    rays_d = intrins_inv @ points
    points1 = depthmap1.reshape(1,-1) * rays_d
    if depthmap2 is not None:
        points2 = depthmap2.reshape(1,-1) * rays_d
        return points1.reshape(3,H,W), points2.reshape(3,H,W)
    else:
        return points1.reshape(3,H,W)


def point_to_normal(view, points1, points2=None):
    points = (
        points1[None] if points2 is None 
        else torch.stack([points1, points2],dim=0)
    )
    output = torch.zeros_like(points)
    dx = points[...,2:, 1:-1] - points[...,:-2, 1:-1]
    dy = points[...,1:-1, 2:] - points[...,1:-1, :-2]
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=1), dim=1)
    output[...,1:-1, 1:-1] = normal_map
    return (
        output[0] if points2 is None 
        else output
    )


def depth_to_normal(view, depth1, depth2=None):
    points = depths_to_points(view, depth1, depth2)
    points = points[None] if depth2 is None else points
    return point_to_normal(view, *points)


def is_in_view_frustum(
    points:torch.Tensor,
    camera:Camera,
) -> torch.Tensor:
    """_summary_

    Args:
        points (torch.Tensor): Tensor with shape (N, 3)
        cameras (List[Camera]): _description_
    """
    H, W = camera.image_height, camera.image_width
    
    view_points = transform_points_world_to_view(
        points.view(1, -1, 3),
        cameras=[camera],
    )[0]  # (N, 3)
    
    pix_pts = transform_points_to_pixel_space(
        view_points.view(1, -1, 3),
        points_are_already_in_view_space=True,
        cameras=[camera],
    )[0]  # (N, 2)
    
    pix_x, pix_y, pix_z = pix_pts[..., 0], pix_pts[..., 1], view_points[..., 2]
    
    valid_mask = (
        (pix_x >= 0) & (pix_x <= W-1) 
        & (pix_y >= 0) & (pix_y <= H-1) 
        & (pix_z > camera.znear) & (pix_z < camera.zfar)
    )  # (N,)
    
    return valid_mask


def unflatten_voronoi_features(voronoi_features:torch.Tensor, n_voronoi_per_gaussians:int=9):
    """Unflatten the voronoi features into a 3D tensor with shape (n_gaussians, n_voronoi_per_gaussians, *voronoi_features.shape[1:]).

    Args:
        voronoi_features (torch.Tensor): Tensor with shape (n_gaussians * n_voronoi_per_gaussians, *voronoi_features.shape[1:]).

    Returns:
        torch.Tensor: Tensor with shape (n_gaussians, n_voronoi_per_gaussians, *voronoi_features.shape[1:]).
    """
    n_gaussians = len(voronoi_features) // n_voronoi_per_gaussians
    return torch.cat(
        [
            voronoi_features[:-n_gaussians].reshape(n_gaussians, n_voronoi_per_gaussians-1, *voronoi_features.shape[1:]),
            voronoi_features[-n_gaussians:].reshape(n_gaussians, 1, *voronoi_features.shape[1:])
        ],
        dim=1
    ).reshape(n_gaussians, n_voronoi_per_gaussians, *voronoi_features.shape[1:])
    
    
def flatten_voronoi_features(voronoi_features:torch.Tensor, n_voronoi_per_gaussians:int=9):
    return torch.cat(
        [
            voronoi_features[:, :n_voronoi_per_gaussians-1].reshape(-1, *voronoi_features.shape[2:]),
            voronoi_features[:, n_voronoi_per_gaussians-1:].reshape(-1, *voronoi_features.shape[2:])
        ],
        dim=0
    )


def identify_out_of_field_points(
    points:torch.Tensor,
    views:List[Camera],
):  
    n_points = points.shape[0]
    out_of_field_mask = torch.ones_like(points[:, 0], dtype=torch.bool, device=points.device)
    
    for camera in tqdm(views, desc="Identifying out of field points"):
        H, W = camera.image_height, camera.image_width

        # Transform points to view space
        view_points = transform_points_world_to_view(
            points=points.view(1, n_points, 3),
            cameras=[camera],
        )[0]  # (N, 3)
        
        # Project points to pixel space
        pix_points = transform_points_to_pixel_space(
            points=view_points.view(1, n_points, 3),
            cameras=[camera],
            points_are_already_in_view_space=True,
            keep_float=True,
        )[0]  # (N, 2)
        int_pix_points = pix_points.round().long()  # (N, 2)
        pix_x, pix_y, pix_z = pix_points[..., 0], pix_points[..., 1], view_points[..., 2]  # (N,)
        int_pix_x, int_pix_y = int_pix_points[..., 0], int_pix_points[..., 1]  # (N,)
        
        # Remove points outside view frustum and outside depth range
        valid_mask = (
            (pix_x >= 0) & (pix_x <= W-1) 
            & (pix_y >= 0) & (pix_y <= H-1) 
        )  # (N,)
        
        out_of_field_mask[valid_mask] = False

    return out_of_field_mask


def compute_face_culling_mask(
    verts: torch.Tensor,
    faces: torch.Tensor,
    camera: Camera,
    near_plane_distance: float = None,
    backface_culling: bool = False,
) -> torch.Tensor:
    """Compute a mask for faces that should be culled based on near-plane and/or back-face criteria.

    Args:
        verts (torch.Tensor): Mesh vertices with shape (N, 3) in world space.
        faces (torch.Tensor): Face indices with shape (F, 3).
        camera (Camera): The camera to use for culling.
        near_plane_distance (float, optional): Minimum distance from camera. Faces with any vertex
            closer than this distance will be culled. If None, no near-plane culling is applied.
        backface_culling (bool): If True, cull faces that face away from the camera.

    Returns:
        torch.Tensor: Boolean mask of shape (F,) where True means the face should be KEPT.
    """
    device = verts.device
    n_faces = faces.shape[0]

    # Start with all faces visible
    keep_mask = torch.ones(n_faces, dtype=torch.bool, device=device)

    if near_plane_distance is not None and near_plane_distance > 0:
        # Transform vertices to view space
        view_verts = transform_points_world_to_view(
            points=verts.unsqueeze(0),
            cameras=[camera],
        )[0]  # (N, 3)

        # Get z-depth (positive = in front of camera)
        vert_depths = view_verts[:, 2]  # (N,)

        # Get depths for each face vertex
        face_depths = vert_depths[faces]  # (F, 3)

        # Cull faces where ANY vertex is closer than the threshold
        min_face_depth = face_depths.min(dim=1)[0]  # (F,)
        near_plane_mask = min_face_depth >= near_plane_distance
        keep_mask = keep_mask & near_plane_mask

    if backface_culling:
        # Get face vertices
        face_verts = verts[faces]  # (F, 3, 3)

        # Compute face normals (cross product of edges)
        v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normals = torch.cross(edge1, edge2, dim=-1)  # (F, 3)
        face_normals = torch.nn.functional.normalize(face_normals, dim=-1)

        # Compute face centers
        face_centers = face_verts.mean(dim=1)  # (F, 3)

        # Compute view direction (from face center to camera)
        camera_pos = camera.camera_center  # (3,)
        view_dirs = camera_pos.unsqueeze(0) - face_centers  # (F, 3)
        view_dirs = torch.nn.functional.normalize(view_dirs, dim=-1)

        # Face is front-facing if normal points toward camera (positive dot product)
        dot_products = (face_normals * view_dirs).sum(dim=-1)  # (F,)
        frontface_mask = dot_products > 0
        keep_mask = keep_mask & frontface_mask

    return keep_mask


def compute_vertex_culling_mask(
    verts: torch.Tensor,
    camera: Camera,
    near_plane_distance: float = None,
) -> torch.Tensor:
    """Compute a mask for vertices that are too close to the camera.

    Args:
        verts (torch.Tensor): Mesh vertices with shape (N, 3) in world space.
        camera (Camera): The camera to use for culling.
        near_plane_distance (float, optional): Minimum distance from camera.

    Returns:
        torch.Tensor: Boolean mask of shape (N,) where True means the vertex should be KEPT.
    """
    if near_plane_distance is None or near_plane_distance <= 0:
        return torch.ones(verts.shape[0], dtype=torch.bool, device=verts.device)

    # Transform vertices to view space
    view_verts = transform_points_world_to_view(
        points=verts.unsqueeze(0),
        cameras=[camera],
    )[0]  # (N, 3)

    # Get z-depth (positive = in front of camera)
    vert_depths = view_verts[:, 2]  # (N,)

    return vert_depths >= near_plane_distance