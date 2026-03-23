#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        # Removed verbose camera loading progress
        # sys.stdout.write('\r')
        # sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        # sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            cx, cy = intr.params[1], intr.params[2]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            cx, cy = intr.params[2], intr.params[3]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        # Warn if principal point is not centered (3DGS assumes centered principal point)
        cx_expected, cy_expected = width / 2.0, height / 2.0
        cx_offset = abs(cx - cx_expected)
        cy_offset = abs(cy - cy_expected)
        # Warn if offset is more than 1% of image dimension
        if cx_offset > width * 0.01 or cy_offset > height * 0.01:
            print(f"\n[WARNING] Camera {idx}: Principal point ({cx:.1f}, {cy:.1f}) is not centered ({cx_expected:.1f}, {cy_expected:.1f})")
            print(f"          Offset: ({cx_offset:.1f}, {cy_offset:.1f}) pixels. 3DGS assumes centered principal point - this may cause misalignment!")

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        # Removed debug message about image dimension mismatches (expected when using downsampled images)
        # actual_w, actual_h = image.size
        # if actual_w != width or actual_h != height:
        #     print(f"\n[DEBUG] Image {image_name}: intrinsics say {width}x{height}, actual image is {actual_w}x{actual_h}")

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    print(f"[INFO] Loading PLY from: {path}")
    print(f"[INFO] Point cloud has {len(positions)} points")
    print(f"[INFO] Available fields: {vertices.data.dtype.names}")

    # Try different color formats
    if 'red' in vertices:
        # Standard PLY format with red/green/blue
        print("[INFO] Using standard RGB color format (red/green/blue)")
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    elif 'f_dc_0' in vertices:
        # Gaussian splatting format with spherical harmonics
        print("[INFO] Using Gaussian splatting SH color format (f_dc_0/f_dc_1/f_dc_2)")
        colors = np.vstack([vertices['f_dc_0'], vertices['f_dc_1'], vertices['f_dc_2']]).T
        # SH coefficients need to be converted (already in 0-1 range typically)
        colors = np.clip(colors, 0, 1)
    else:
        # No color information, use white
        print("[WARNING] No color fields found, using white for all points")
        colors = np.ones_like(positions)

    # Handle normals
    if 'nx' in vertices:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
        print("[INFO] Normals found in PLY file")
    elif 'rot_0' in vertices or 'rotation_0' in vertices:
        # Compute normals from Gaussian quaternion rotations
        # The Z-axis of the rotation matrix is the surface normal
        print("[INFO] Computing normals from Gaussian quaternion rotations (Z-axis of rotation matrix)")

        # Detect quaternion field names (could be rot_X or rotation_X)
        if 'rot_0' in vertices:
            quat_fields = ['rot_0', 'rot_1', 'rot_2', 'rot_3']
        else:
            quat_fields = ['rotation_0', 'rotation_1', 'rotation_2', 'rotation_3']

        # Load quaternions (stored as w,x,y,z in 3DGS/gsplat PLY files)
        quats = np.vstack([vertices[field] for field in quat_fields]).T  # (N, 4)

        # Convert quaternions to rotation matrices and extract Z-axis
        normals = np.zeros_like(positions)
        for i, q in enumerate(quats):
            # Quaternion to rotation matrix (WXYZ format, same as COLMAP)
            w, x, y, z = q

            # Rotation matrix from quaternion
            R = np.array([
                [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
                [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
                [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
            ])

            # Z-axis (third column) is the normal, normalize for safety
            normal = R[:, 2]
            normals[i] = normal / np.linalg.norm(normal)

        print(f"[INFO] Computed {len(normals)} normals from quaternions")
    else:
        normals = np.zeros_like(positions)
        print("[WARNING] No normals or quaternions found in PLY file, using zeros")

    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, init_ply_path=None, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # Handle init.ply with optional override
    if init_ply_path and init_ply_path != "":
        # User specified override path
        if not os.path.isabs(init_ply_path):
            init_ply_path = os.path.join(path, init_ply_path)
        print(f"[INFO] Using user-specified init.ply: {init_ply_path}")
        ply_path = init_ply_path
        pcd = fetchPly(ply_path)
    elif os.path.exists(os.path.join(path, "sparse/0/init.ply")):
        # Check for init.ply in default location (gsplat format)
        ply_path = os.path.join(path, "sparse/0/init.ply")
        print(f"[INFO] Using init.ply from default location: {ply_path}")
        pcd = fetchPly(ply_path)
    else:
        # Fallback to standard COLMAP points3D files
        print("[INFO] No init.ply found, using COLMAP points3D files")
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")

        xyz, rgb, _ = read_points3D_binary(bin_path)
        storePly(ply_path, xyz, rgb)
        pcd = fetchPly(ply_path)

    # Debug: Compare spatial extent of point cloud vs camera positions
    from utils.graphics_utils import getWorld2View2
    pcd_center = pcd.points.mean(axis=0)
    pcd_extent = pcd.points.max(axis=0) - pcd.points.min(axis=0)
    cam_centers = []
    for cam in train_cam_infos:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3])
    cam_centers = np.array(cam_centers)
    cam_center = cam_centers.mean(axis=0)
    cam_extent = cam_centers.max(axis=0) - cam_centers.min(axis=0)
    print(f"\n[DEBUG] Point cloud center: {pcd_center}")
    print(f"[DEBUG] Point cloud extent: {pcd_extent}")
    print(f"[DEBUG] Camera center: {cam_center}")
    print(f"[DEBUG] Camera extent: {cam_extent}")
    print(f"[DEBUG] Distance between centers: {np.linalg.norm(pcd_center - cam_center):.4f}")
    print(f"[DEBUG] NeRF++ normalization radius: {nerf_normalization['radius']:.4f}")

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}