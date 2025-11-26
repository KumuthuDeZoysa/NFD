"""
Point Cloud Dataset Generation Module

Converts 3D meshes to point clouds with various augmentations:
- Uniform/random surface sampling
- Sensor noise simulation (Gaussian, uniform, distance-dependent)
- Partial view generation (occlusion simulation)
- Multiple output formats (PLY, PCD, XYZ, NPY)

Usage:
    import mesh_to_pointcloud as m2pc
    
    # Basic conversion
    points, normals = m2pc.mesh_to_pointcloud('mesh.obj', num_points=2048)
    m2pc.save_pointcloud(points, normals, 'output.ply')
    
    # Full pipeline with variants
    m2pc.process_mesh_to_dataset('mesh.obj', 'output_dir/', num_variants=5)
"""

import trimesh
import numpy as np
import open3d as o3d
import os
from pathlib import Path


def mesh_to_pointcloud(mesh_path, num_points=2048, method='uniform'):
    """
    Convert mesh to point cloud.
    
    Args:
        mesh_path: Path to .obj mesh file
        num_points: Number of points to sample
        method: 'uniform' or 'random'
    
    Returns:
        points: (N, 3) array of point coordinates
        normals: (N, 3) array of surface normals
    """
    # Load mesh
    mesh = trimesh.load(mesh_path, force='mesh')
    
    if method == 'uniform':
        # Uniform sampling (better distribution)
        points, face_indices = trimesh.sample.sample_surface_even(mesh, num_points)
    else:
        # Random sampling (faster)
        points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
    
    # Compute normals at sampled points
    normals = mesh.face_normals[face_indices]
    
    return points, normals


def add_sensor_noise(points, noise_type='gaussian', noise_level=0.01):
    """
    Add realistic sensor noise to simulate LiDAR/depth cameras.
    
    Args:
        points: (N, 3) point coordinates
        noise_type: 'gaussian', 'uniform', or 'distance_dependent'
        noise_level: Standard deviation for gaussian noise
    
    Returns:
        noisy_points: (N, 3) with added noise
    """
    if noise_type == 'gaussian':
        # Standard gaussian noise
        noise = np.random.normal(0, noise_level, points.shape)
        return points + noise
    
    elif noise_type == 'uniform':
        # Uniform random noise
        noise = np.random.uniform(-noise_level, noise_level, points.shape)
        return points + noise
    
    elif noise_type == 'distance_dependent':
        # Noise increases with distance from origin (realistic LiDAR)
        distances = np.linalg.norm(points, axis=1, keepdims=True)
        noise_scale = noise_level * (1 + distances / (distances.max() + 1e-8))
        noise = np.random.normal(0, 1, points.shape) * noise_scale
        return points + noise
    
    return points


def generate_partial_view(points, normals, view_point=None, occlusion_ratio=0.5):
    """
    Generate partial point cloud (simulate occlusion/single view).
    
    Args:
        points: (N, 3) complete point cloud
        normals: (N, 3) surface normals
        view_point: (3,) camera position, if None uses random
        occlusion_ratio: fraction of points to keep (0.5 = half visible)
    
    Returns:
        partial_points: Subset of visible points
        partial_normals: Corresponding normals
    """
    if view_point is None:
        # Random viewpoint on sphere
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        radius = 2.0
        view_point = np.array([
            radius * np.sin(phi) * np.cos(theta),
            radius * np.sin(phi) * np.sin(theta),
            radius * np.cos(phi)
        ])
    
    # Compute vectors from viewpoint to each point
    view_vectors = points - view_point
    view_vectors = view_vectors / (np.linalg.norm(view_vectors, axis=1, keepdims=True) + 1e-8)
    
    # Keep points facing the viewpoint (dot product > 0)
    visibility = np.sum(normals * (-view_vectors), axis=1) > 0
    
    # Apply occlusion ratio
    visible_indices = np.where(visibility)[0]
    if len(visible_indices) == 0:
        # Fallback: return random subset if no points are visible
        num_keep = int(len(points) * occlusion_ratio)
        selected = np.random.choice(len(points), num_keep, replace=False)
        return points[selected], normals[selected]
    
    num_keep = int(len(visible_indices) * occlusion_ratio)
    num_keep = max(1, num_keep)  # Ensure at least 1 point
    selected = np.random.choice(visible_indices, num_keep, replace=False)
    
    return points[selected], normals[selected]


def normalize_pointcloud(points, method='sphere'):
    """
    Normalize point cloud to unit sphere or cube.
    
    Args:
        points: (N, 3) array
        method: 'sphere' or 'cube'
    
    Returns:
        normalized_points: (N, 3) normalized to [-1, 1]
    """
    # Center at origin
    points = points - points.mean(axis=0)
    
    if method == 'sphere':
        # Scale to unit sphere
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 1e-8:
            points = points / max_dist
    elif method == 'cube':
        # Scale to unit cube
        max_coord = np.abs(points).max()
        if max_coord > 1e-8:
            points = points / max_coord
    
    return points


def save_pointcloud(points, normals, output_path, format='ply'):
    """
    Save point cloud to file.
    
    Args:
        points: (N, 3) coordinates
        normals: (N, 3) normals
        output_path: Save path
        format: 'ply', 'pcd', 'xyz', or 'npy'
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    if format == 'npy':
        # NumPy format (includes normals)
        np.save(output_path, {'points': points, 'normals': normals})
    elif format == 'xyz':
        # Simple XYZ format (no normals)
        np.savetxt(output_path, points, fmt='%.6f')
    else:
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        
        # Save in requested format
        if format == 'ply':
            o3d.io.write_point_cloud(output_path, pcd)
        elif format == 'pcd':
            o3d.io.write_point_cloud(output_path, pcd)


def process_mesh_to_dataset(
    mesh_path,
    output_dir,
    num_points=2048,
    num_variants=5,
    add_noise=True,
    generate_partial=True
):
    """
    Complete pipeline: mesh → multiple point cloud variants.
    
    Args:
        mesh_path: Input mesh file
        output_dir: Where to save point clouds
        num_points: Points per cloud
        num_variants: How many variants to generate
        add_noise: Whether to add sensor noise
        generate_partial: Whether to create partial views
    """
    os.makedirs(output_dir, exist_ok=True)
    mesh_name = Path(mesh_path).stem
    
    try:
        # Base conversion
        points, normals = mesh_to_pointcloud(mesh_path, num_points)
        points = normalize_pointcloud(points)
        
        # Save complete clean version
        save_pointcloud(
            points, normals,
            f"{output_dir}/{mesh_name}_complete.ply"
        )
        
        # Generate variants
        for i in range(num_variants):
            variant_points = points.copy()
            variant_normals = normals.copy()
            
            if add_noise:
                # Add different noise types
                noise_types = ['gaussian', 'uniform', 'distance_dependent']
                noise_type = noise_types[i % len(noise_types)]
                variant_points = add_sensor_noise(
                    variant_points,
                    noise_type=noise_type,
                    noise_level=0.01
                )
            
            if generate_partial:
                # Create partial view
                variant_points, variant_normals = generate_partial_view(
                    variant_points,
                    variant_normals,
                    occlusion_ratio=np.random.uniform(0.4, 0.7)
                )
            
            # Save variant
            save_pointcloud(
                variant_points,
                variant_normals,
                f"{output_dir}/{mesh_name}_variant_{i}.ply"
            )
        
        print(f"  ✓ Generated {num_variants + 1} point clouds for {mesh_name}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {mesh_name}: {str(e)}")
        return False


def generate_multiview_pointclouds(mesh_path, num_views=6, num_points=2048):
    """
    Generate point clouds from multiple consistent viewpoints.
    
    Args:
        mesh_path: Path to mesh file
        num_views: Number of views to generate
        num_points: Points per cloud
    
    Returns:
        views: List of (points, normals) tuples
    """
    views = []
    angles = np.linspace(0, 2*np.pi, num_views, endpoint=False)
    
    # Load and convert mesh once
    points, normals = mesh_to_pointcloud(mesh_path, num_points)
    points = normalize_pointcloud(points)
    
    for angle in angles:
        view_point = np.array([
            2.0 * np.cos(angle),
            2.0 * np.sin(angle),
            1.0
        ])
        partial, partial_n = generate_partial_view(points, normals, view_point, occlusion_ratio=0.6)
        views.append((partial, partial_n))
    
    return views


def filter_quality_pointclouds(points, min_points=512):
    """
    Filter out low-quality point clouds.
    
    Args:
        points: (N, 3) point array
        min_points: Minimum number of points required
    
    Returns:
        points if quality check passes, None otherwise
    """
    if len(points) < min_points:
        return None
    
    # Check for degeneracy
    variance = np.var(points, axis=0)
    if np.any(variance < 1e-6):
        return None
    
    return points


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert meshes to point clouds')
    parser.add_argument('--input', type=str, required=True,
                        help='Input mesh file or directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for point clouds')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='Number of points to sample')
    parser.add_argument('--num_variants', type=int, default=5,
                        help='Number of variants per mesh')
    parser.add_argument('--no_noise', action='store_true',
                        help='Disable noise addition')
    parser.add_argument('--no_partial', action='store_true',
                        help='Disable partial view generation')
    
    args = parser.parse_args()
    
    # Check if input is file or directory
    if os.path.isfile(args.input):
        # Single file
        process_mesh_to_dataset(
            args.input,
            args.output,
            num_points=args.num_points,
            num_variants=args.num_variants,
            add_noise=not args.no_noise,
            generate_partial=not args.no_partial
        )
    elif os.path.isdir(args.input):
        # Directory of meshes
        mesh_files = [f for f in os.listdir(args.input) if f.endswith(('.obj', '.ply', '.stl'))]
        print(f"Found {len(mesh_files)} mesh files")
        
        for mesh_file in mesh_files:
            mesh_path = os.path.join(args.input, mesh_file)
            mesh_name = Path(mesh_file).stem
            output_dir = os.path.join(args.output, f"object_{mesh_name}")
            
            process_mesh_to_dataset(
                mesh_path,
                output_dir,
                num_points=args.num_points,
                num_variants=args.num_variants,
                add_noise=not args.no_noise,
                generate_partial=not args.no_partial
            )
    else:
        print(f"Error: {args.input} is not a valid file or directory")
