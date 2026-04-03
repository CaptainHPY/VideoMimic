import numpy as np
import torch
from numpy.random import choice
import glob
import os
import trimesh
from tqdm import tqdm
from typing import Optional
import random

SHARED_FLAT_TERRAIN_TOKEN = '__shared_flat_terrain__'


def create_flat_plane_mesh(size=20.0, z=0.0):
    """Create a simple flat plane mesh centered at origin."""
    half = size * 0.5
    vertices = np.array([
        [-half, -half, z],
        [half, -half, z],
        [half, half, z],
        [-half, half, z],
    ], dtype=np.float32)
    triangles = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=np.uint32)
    return vertices, triangles

def generate_terrain_noise(vertices, base_frequency=0.2, amplitude=0.1, octaves=3, persistence=0.5):
    """
    Generate smooth, multi-scale noise for terrain height variation using numpy.
    
    Args:
        vertices: Array of vertex positions
        base_frequency: Base frequency of the noise (in meters)
        amplitude: Maximum height variation
        octaves: Number of noise layers to combine
        persistence: How much each octave contributes (0-1)
    
    Returns:
        Array of height offsets for each vertex
    """
    height_offsets = np.zeros(len(vertices))
    
    # Generate random offsets for each octave
    offsets = np.random.rand(octaves, 2) * 1000  # Large random offset to avoid patterns
    
    for i, (x, y, z) in enumerate(vertices):
        total = 0
        frequency = base_frequency
        amplitude_current = amplitude
        
        for octave in range(octaves):
            # Generate smooth noise using sine waves with random phase
            nx = x * frequency + offsets[octave, 0]
            ny = y * frequency + offsets[octave, 1]
            
            # Combine multiple sine waves for more natural look
            noise = (np.sin(nx) + np.sin(ny) + np.sin(nx + ny)) / 3.0
            
            # Add some randomness to the frequency for each octave
            freq_variation = random.uniform(0.98, 1.02)
            total += noise * amplitude_current * freq_variation
            
            frequency *= 2
            amplitude_current *= persistence
        
        height_offsets[i] = total
    
    # Ensure zero mean by subtracting the average
    height_offsets -= np.mean(height_offsets)
    
    return height_offsets

def convert_mesh_to_heightfield(mesh_file, resolution):
    """
    Converts a mesh file into a heightfield trimesh using ray casting.

    Args:
        mesh_file (str): Path to the mesh file.
        resolution (float): The resolution for the heightfield grid (e.g., 0.01 for 1cm).

    Returns:
        tuple: (vertices, triangles) for the heightfield mesh.
    """
    original_mesh = trimesh.load(mesh_file)
    original_tri_count = len(original_mesh.faces)
    original_vert_count = len(original_mesh.vertices)

    # --- Start Heightfield Conversion ---
    bounds = original_mesh.bounds
    min_bound, max_bound = bounds

    # Add a small buffer to capture edges and ensure rays start above
    buffer = resolution * 2
    min_bound_buffered = min_bound - buffer
    max_bound_buffered = max_bound + buffer

    # Calculate grid dimensions based on buffered bounds
    x_range = max_bound_buffered[0] - min_bound_buffered[0]
    y_range = max_bound_buffered[1] - min_bound_buffered[1]
    x_steps = int(np.ceil(x_range / resolution))
    y_steps = int(np.ceil(y_range / resolution))

    # Ensure we have at least 2x2 steps for triangulation
    if x_steps < 2: x_steps = 2
    if y_steps < 2: y_steps = 2

    # Create grid points (centers of cells)
    x_coords = np.linspace(min_bound_buffered[0] + resolution / 2, max_bound_buffered[0] - resolution / 2, x_steps)
    y_coords = np.linspace(min_bound_buffered[1] + resolution / 2, max_bound_buffered[1] - resolution / 2, y_steps)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)

    # Ray origins: Place slightly above the highest point of the buffered bounds
    ray_origins = np.zeros((x_steps * y_steps, 3))
    ray_origins[:, 0] = grid_x.ravel()
    ray_origins[:, 1] = grid_y.ravel()
    ray_origins[:, 2] = max_bound_buffered[2] # Start rays above the mesh

    # Ray directions: Straight down
    ray_directions = np.tile([0, 0, -1], (x_steps * y_steps, 1))

    # Cast rays
    locations, index_ray, _ = original_mesh.ray.intersects_location(
        ray_origins=ray_origins, ray_directions=ray_directions
    )

    # Initialize height map with a default low value (e.g., buffered min z bound)
    min_z_original = min_bound[2]
    height_map = np.full(grid_x.shape, min_z_original - buffer, dtype=np.float32)

    # Fill height map with intersection points
    if len(index_ray) > 0:
        hit_indices = np.unravel_index(index_ray, grid_x.shape)
        height_map[hit_indices] = locations[:, 2] # Z coordinate of intersection

    # --- Create Trimesh from Heightfield ---
    vertices_list = []
    faces_list = []
    vertex_map = {} # Map (row, col) to vertex index

    # Create vertices
    vert_idx = 0
    for r in range(y_steps):
        for c in range(x_steps):
            vertex = [x_coords[c], y_coords[r], height_map[r, c]]
            vertices_list.append(vertex)
            vertex_map[(r, c)] = vert_idx
            vert_idx += 1

    # Create faces (two triangles per grid cell)
    for r in range(y_steps - 1):
        for c in range(x_steps - 1):
            idx00 = vertex_map[(r, c)]
            idx10 = vertex_map[(r, c + 1)]
            idx01 = vertex_map[(r + 1, c)]
            idx11 = vertex_map[(r + 1, c + 1)]
            faces_list.append([idx00, idx10, idx11])
            faces_list.append([idx00, idx11, idx01])

    vertices = np.array(vertices_list, dtype=np.float32)
    triangles = np.array(faces_list, dtype=np.uint32)

    heightfield_tri_count = len(triangles)
    heightfield_vert_count = len(vertices)
    print(f"Mesh: {os.path.basename(mesh_file)} - Original Verts/Tris: {original_vert_count}/{original_tri_count}, Heightfield Verts/Tris: {heightfield_vert_count}/{heightfield_tri_count}")

    if len(vertices) == 0 or len(triangles) == 0:
         print(f"Warning: Heightfield conversion for {mesh_file} resulted in empty mesh. Using original.")
         vertices = np.array(original_mesh.vertices, dtype=np.float32)
         triangles = np.array(original_mesh.faces, dtype=np.uint32)
    # --- End Heightfield Conversion ---

    return vertices, triangles

def load_all_meshes(mesh_files, convert_to_heightfield: bool = False, cfg=None):
    """
    Load all meshes matching the given glob pattern. Optionally converts them to heightfields.
    Uses a cache to avoid reprocessing identical mesh files.

    Args:
        mesh_files (list): List of mesh files.
        convert_to_heightfield (bool): If True, convert meshes to heightfields.
        cfg: Terrain config for flat fallback options.

    Returns:
        list: List of tuples (vertices, triangles) for each mesh (original or heightfield).
    """
    meshes = []
    mesh_cache = {} # Cache for processed mesh data (vertices, triangles)
    resolution = 0.025  # 1cm resolution for heightfield conversion

    action = "converting to heightfields" if convert_to_heightfield else "loading"
    print(f"Processing {len(mesh_files)} unique mesh files ({action})...")
    print(mesh_files)

    flat_size = float(getattr(cfg, 'flat_terrain_size', 20.0))
    flat_z = float(getattr(cfg, 'flat_terrain_z', 0.0))
    auto_flat_if_missing = bool(getattr(cfg, 'auto_flat_if_mesh_missing', False))

    for mesh_file in tqdm(mesh_files):
        if mesh_file in mesh_cache:
            # Load from cache
            vertices, triangles = mesh_cache[mesh_file]
        else:
            # Process mesh
            if mesh_file == SHARED_FLAT_TERRAIN_TOKEN:
                print(f"Using shared generated flat terrain mesh: size={flat_size}, z={flat_z}")
                vertices, triangles = create_flat_plane_mesh(size=flat_size, z=flat_z)
            elif convert_to_heightfield:
                print(f"Converting {mesh_file} to heightfield...")
                vertices, triangles = convert_mesh_to_heightfield(mesh_file, resolution)
            else:
                if not os.path.exists(mesh_file):
                    if auto_flat_if_missing:
                        print(
                            f"Terrain mesh not found ({mesh_file}), "
                            f"using generated flat terrain size={flat_size}, z={flat_z}."
                        )
                        vertices, triangles = create_flat_plane_mesh(size=flat_size, z=flat_z)
                    else:
                        raise FileNotFoundError(
                            f"Terrain mesh not found: {mesh_file}. "
                            "Set --env.terrain.auto_flat_if_mesh_missing=True to auto-generate flat terrain."
                        )
                else:
                    print(f"Loading original mesh {mesh_file}...")
                    mesh = trimesh.load(mesh_file)
                    vertices = np.array(mesh.vertices, dtype=np.float32)
                    triangles = np.array(mesh.faces, dtype=np.uint32)
                    print(f"Mesh: {os.path.basename(mesh_file)} - Original Tris: {len(triangles)}")

            # Store in cache
            mesh_cache[mesh_file] = (vertices, triangles)

        meshes.append((vertices, triangles))

    return meshes

def duplicate_mesh_grid_multi(meshes, n_rows, noise_config=None):
    """
    Arrange multiple meshes in a grid pattern with specified number of rows.
    Number of columns is determined by number of meshes.
    
    Args:
        meshes: list of (vertices, triangles) tuples for each mesh
        n_rows: number of rows in the grid
        noise_config: dictionary containing noise parameters:
            - base_frequency: base frequency of the noise (default: 0.2)
            - amplitude: maximum height variation (default: 0.1)
            - octaves: number of noise layers (default: 3)
            - persistence: how much each octave contributes (default: 0.5)
        
    Returns:
        new_vertices: numpy array of duplicated vertices
        new_triangles: numpy array of duplicated face indices
    """
    n_meshes = len(meshes)
    if n_meshes == 0:
        raise ValueError(
            "No terrain meshes were loaded. Check replay data / terrain path configuration and ensure matching files exist."
        )
    n_cols = n_meshes  # One column per unique mesh
    
    # Calculate the maximum dimensions across all meshes
    max_width = 0
    max_depth = 0
    for vertices, _ in meshes:
        min_coords = vertices.min(axis=0)
        max_coords = vertices.max(axis=0)
        width = max_coords[0] - min_coords[0]
        depth = max_coords[1] - min_coords[1]
        max_width = max(max_width, width)
        max_depth = max(max_depth, depth)
    
    # Add small gap between duplicated meshes
    width_spacing = max_width #* 1.05
    depth_spacing = max_depth #* 1.05
    
    # Initialize lists to store new vertices and triangles
    new_vertices = []
    new_triangles = []
    vertex_count = 0
    
    
    # Duplicate meshes for each grid position
    print(f"Arranging meshes in a {n_rows}x{n_cols} grid...")
    for row in tqdm(range(n_rows)):
        for col in range(n_cols):
            # Get the mesh for this column
            vertices, triangles = meshes[col]

            if noise_config is not None and noise_config.get('random_z_scaling_enable', False):
                min_hight = vertices[:, 2].min()

                random_add = (np.random.rand() - 0.5) * 2.0

                scaling_factor = 1 + random_add * noise_config.get('random_z_scaling_scale', 0.0)

                vertices[:, 2] = (vertices[:, 2] - min_hight) * scaling_factor + min_hight
            

            
            # Calculate offset for this grid position
            offset = np.array([
                col * width_spacing,
                row * depth_spacing,
                0
            ], dtype=np.float32)
            
            # Duplicate and translate vertices
            curr_vertices = vertices + offset
            
            if noise_config is not None and col % 2 == 0 and noise_config.get('amplitude', 0.0) > 0:
                # Apply terrain noise
                height_offsets = generate_terrain_noise(
                    curr_vertices,
                    base_frequency=noise_config.get('base_frequency', 0.2),
                    amplitude=noise_config.get('amplitude', 0.1),
                    octaves=noise_config.get('octaves', 3),
                    persistence=noise_config.get('persistence', 0.5)
                )
                curr_vertices[:, 2] += height_offsets
            
            # Duplicate triangles with updated indices
            curr_triangles = triangles + vertex_count
            
            # Update vertex count
            vertex_count += len(vertices)
            
            # Add to our lists
            new_vertices.append(curr_vertices)
            new_triangles.append(curr_triangles)
    
    # Combine all vertices and triangles
    new_vertices = np.vstack(new_vertices).astype(np.float32)
    new_triangles = np.vstack(new_triangles).astype(np.uint32)
    
    return new_vertices, new_triangles, width_spacing, depth_spacing

def get_terrain_offset(terrain_idx: torch.Tensor, width_spacing: float, depth_spacing: float, n_rows: int = 1) -> torch.Tensor:
    """
    Vectorized function to get offsets for terrain indices with optional random row placement.
    
    Args:
        terrain_idx: Tensor of terrain indices
        width_spacing: Spacing between terrains in x direction
        depth_spacing: Spacing between terrains in y direction
        n_rows: Number of rows for random placement
        
    Returns:
        Tensor of shape (N, 3) containing [x, y, z] offsets for each terrain
    """
    if n_rows > 1:
        row_idx = torch.randint(0, n_rows, terrain_idx.shape, device=terrain_idx.device)
    else:
        row_idx = torch.zeros_like(terrain_idx)
    
    x_offset = terrain_idx * width_spacing
    y_offset = row_idx * depth_spacing
    z_offset = torch.zeros_like(x_offset)
    
    return torch.stack([x_offset, y_offset, z_offset], dim=-1)

class DeepMimicTerrain:
    def __init__(self, cfg, num_robots, terrain_paths) -> None:
        self.cfg = cfg
        # Determine if heightfield conversion is needed from cfg
        convert_to_heightfield = getattr(cfg, 'cast_mesh_to_heightfield', False) # Default to False if not specified
        self.meshes = load_all_meshes(terrain_paths, convert_to_heightfield=convert_to_heightfield, cfg=cfg)
        self.n_terrains = len(self.meshes)

        # Calculate max dimensions for spacing
        max_width = 0
        max_depth = 0
        for vertices, _ in self.meshes:
            min_coords = vertices.min(axis=0)
            max_coords = vertices.max(axis=0)
            width = max_coords[0] - min_coords[0]
            depth = max_coords[1] - min_coords[1]
            max_width = max(max_width, width)
            max_depth = max(max_depth, depth)

        self.n_rows = cfg.n_rows
        
        # Get noise configuration from cfg
        noise_config = getattr(cfg, 'terrain_noise', {}) # Use empty dict if not present
        
        # Create grid with specified number of rows and columns equal to number of meshes
        self.vertices, self.triangles, self.width_spacing, self.depth_spacing = duplicate_mesh_grid_multi(
            self.meshes, 
            cfg.n_rows,
            noise_config=noise_config
        )

    def get_terrain_offset(self, terrain_idx: int) -> np.ndarray:
        """
        Get the offset for a specific terrain with optional random row placement.
        
        Args:
            terrain_idx (torch.Tensor): Tensor of terrain indices
        Returns:
            np.ndarray: [x, y, z] offset for the specified terrain
        """
        if torch.any(terrain_idx >= self.n_terrains):
            raise ValueError(f"Terrain index {terrain_idx} exceeds number of available terrains ({self.n_terrains})")
            
        return get_terrain_offset(
            terrain_idx=terrain_idx,
            width_spacing=self.width_spacing,
            depth_spacing=self.depth_spacing,
            n_rows=self.n_rows,
        )