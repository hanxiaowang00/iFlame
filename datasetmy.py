import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import networkx as nx
import json
import trimesh



def get_shifted_sequence(sequence):
    non_special = np.flatnonzero(np.isin(sequence, [0, 1, 2], invert=True))
    if non_special.shape[0] > 0:
        idx = non_special[0]
        val = sequence[idx]
        sequence[non_special] -= (val - 3)
    return sequence


def read_faces(text):
    all_lines = text.splitlines()
    all_face_lines = [x for x in all_lines if x.startswith('f ')]
    all_faces = [[int(y.split('/')[0]) - 1 for y in x.strip().split(' ')[1:]] for x in all_face_lines]
    return all_faces


def read_vertices(text):
    all_lines = text.splitlines()
    all_vertex_lines = [x for x in all_lines if x.startswith('v ')]
    all_vertices = np.array([[float(y) for y in x.strip().split(' ')[1:]] for x in all_vertex_lines])
    assert all_vertices.shape[1] == 3, 'vertices should have 3 coordinates'
    return all_vertices


def quantize_coordinates(coords, num_tokens=256):
    if torch.is_tensor(coords):
        coords = torch.clip((coords + 0.5), 0, 1) * num_tokens  # type: ignore
        coords_quantized = coords.round().long()
    else:
        coords = np.clip((coords + 0.5), 0, 1) * num_tokens  # type: ignore
        coords_quantized = coords.round().astype(int)
    return coords_quantized


def face_to_cycles(face):
    """Find cycles in face."""
    g = nx.Graph()
    for v in range(len(face) - 1):
        g.add_edge(face[v], face[v + 1])
    g.add_edge(face[-1], face[0])
    return list(nx.cycle_basis(g))


def sort_vertices_and_faces(vertices_, faces_, num_tokens=256):
    vertices = np.clip((vertices_ + 0.5), 0, 1) * num_tokens  # type: ignore
    vertices_quantized_ = vertices.round().astype(int)

    vertices_quantized_ = vertices_quantized_[:, [2, 0, 1]]
    vertices_quantized, unique_inverse = np.unique(vertices_quantized_, axis=0, return_inverse=True)

    sort_inds = np.lexsort(vertices_quantized.T)

    vertices_quantized = vertices_quantized[sort_inds]
    vertices_quantized = np.stack([vertices_quantized[:, 2], vertices_quantized[:, 1], vertices_quantized[:, 0]], axis=-1)

    # Re-index faces and tris to re-ordered vertices.
    faces = [np.argsort(sort_inds)[unique_inverse[f]] for f in faces_]
    # Merging duplicate vertices and re-indexing the faces causes some faces to
    # contain loops (e.g [2, 3, 5, 2, 4]). Split these faces into distinct
    # sub-faces.
    sub_faces = []
    for f in faces:
        cliques = face_to_cycles(f)
        for c in cliques:
            c_length = len(c)
            # Only append faces with more than two verts.
            if c_length > 2:
                d = np.argmin(c)
                # Cyclically permute faces just that first index is the smallest.
                sub_faces.append([c[(d + i) % c_length] for i in range(c_length)])
    faces = sub_faces
    # Sort faces by lowest vertex indices. If two faces have the same lowest
    # index then sort by next lowest and so on.
    faces.sort(key=lambda f: tuple(sorted(f)))

    # After removing degenerate faces some vertices are now unreferenced.
    # Remove these.
    num_verts = vertices_quantized.shape[0]
    vert_connected = np.equal(
        np.arange(num_verts)[:, None], np.hstack(faces)[None]).any(axis=-1)
    vertices_quantized = vertices_quantized[vert_connected]
    # Re-index faces and tris to re-ordered vertices.
    vert_indices = (
            np.arange(num_verts) - np.cumsum(1 - vert_connected.astype('int')))
    faces = [vert_indices[f].tolist() for f in faces]
    vertices = vertices_quantized / num_tokens - 0.5
    # order: Z, Y, X --> X, Y, Z
    vertices = np.stack([vertices[:, 2], vertices[:, 1], vertices[:, 0]], axis=-1)
    return vertices, faces

def scale_vertices(vertices, x_lims=(0.75, 1.25), y_lims=(0.75, 1.25), z_lims=(0.75, 1.25)):
    # scale x, y, z
    x = np.random.uniform(low=x_lims[0], high=x_lims[1], size=(1,))
    y = np.random.uniform(low=y_lims[0], high=y_lims[1], size=(1,))
    z = np.random.uniform(low=z_lims[0], high=z_lims[1], size=(1,))
    vertices = np.stack([vertices[:, 0] * x, vertices[:, 1] * y, vertices[:, 2] * z], axis=-1)
    return vertices


def shift_vertices(vertices, x_lims=(-0.1, 0.1), y_lims=(-0.1, 0.1), z_lims=(-0.075, 0.075)):
    # shift x, y, z
    x = np.random.uniform(low=x_lims[0], high=x_lims[1], size=(1,))
    y = np.random.uniform(low=y_lims[0], high=y_lims[1], size=(1,))
    z = np.random.uniform(low=z_lims[0], high=z_lims[1], size=(1,))
    x = max(min(x, 0.5 - vertices[:, 0].max()), -0.5 - vertices[:, 0].min())
    y = max(min(y, 0.5 - vertices[:, 1].max()), -0.5 - vertices[:, 1].min())
    z = max(min(z, 0.5 - vertices[:, 2].max()), -0.5 - vertices[:, 2].min())
    vertices = np.stack([vertices[:, 0] + x, vertices[:, 1] + y, vertices[:, 2] + z], axis=-1)
    return vertices


def normalize_vertices(vertices):
    bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])  # type: ignore
    vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
    vertices = vertices / (bounds[1] - bounds[0]).max()
    return vertices


def create_feature_stack(vertices, faces, num_tokens):
    vertices, faces = sort_vertices_and_faces(vertices, faces, num_tokens)
    # need more features: positions, angles, area, cross_product
    triangles = vertices[faces, :]
    triangles= create_feature_stack_from_triangles(triangles)
    return triangles, vertices, faces


def create_feature_stack_from_triangles(triangles):
    # t_areas = area(triangles) * 1e3
    # t_angles = angle(triangles) / float(np.pi)
    # t_normals = unit_vector(normal(triangles))
    return triangles.reshape(-1, 9)#, t_normals.reshape(-1, 3), t_areas.reshape(-1, 1), t_angles.reshape(-1, 3)

class TrianglesDataset(Dataset):

    def __init__(self, dataset_path='processed_data.pkl', split='train', scale_augment=True, shift_augment=True,
                 overfit=False, num_tokens=256,category_prefix=None,allmaxlen=0):
    
        self.dataset_path = Path(dataset_path)
        self.scale_augment = scale_augment
        self.shift_augment = shift_augment
        self.num_tokens = num_tokens
        self.overfit = overfit

        self.cached_vertices = []
        self.cached_faces = []
        self.names = []
        self.split=split
        self.category_prefix=category_prefix

        with open(self.dataset_path, 'rb') as f:
            data = pickle.load(f)

            if self.category_prefix:
                filtered_names = []
                filtered_vertices = []
                filtered_faces = []
                for idx, name in enumerate(data[f'name_{self.split}']):
                    if name.startswith(self.category_prefix):
                        filtered_names.append(name)
                        filtered_vertices.append(data[f'vertices_{self.split}'][idx])
                        filtered_faces.append(data[f'faces_{self.split}'][idx])
                self.names = filtered_names
                self.cached_vertices = filtered_vertices
                self.cached_faces = filtered_faces
            else:
                # Load all categories if no specific prefix is provided
                self.names = data[f'name_{self.split}']
                self.cached_vertices = data[f'vertices_{self.split}']
                self.cached_faces = data[f'faces_{self.split}']

            # Handling overfitting scenario
            if overfit:
                multiplier = 1 if self.split == 'val' else 500
                self.names = data['name_train'][:1] * multiplier
                self.cached_vertices = data['vertices_train'][:1] * multiplier
                self.cached_faces = data['faces_train'][:1] * multiplier
            
        print(f"{len(self.cached_vertices)} meshes loaded. 767faces")


        self.maxlen=767+1+1
        self.allmaxlen=0#allmaxlen-self.maxlen
    def __len__(self):
        return len(self.names)

    def __getitem__(self, mesh_idx):

        vertices = self.cached_vertices[mesh_idx]
        faces = self.cached_faces[mesh_idx]
        vertices = normalize_vertices(vertices)
        # 每次 __getitem__ 时实时进行数据增强：
        if self.scale_augment:
            vertices = scale_vertices(vertices)
        vertices = normalize_vertices(vertices)
        if self.shift_augment:
            vertices = shift_vertices(vertices)
        

        triangle_feature, *_ = create_feature_stack(vertices, faces, self.num_tokens)

        mesh_data = quantize_coordinates(torch.tensor(triangle_feature),self.num_tokens)#[:9]
        current_length = mesh_data.shape[0]
    
        # Calculate how many rows to pad
        padding_needed = self.maxlen - current_length-2
        padding = np.ones((1, mesh_data.shape[1])) *(self.num_tokens +2) # pad with zeros, shape (padding_needed, 9)
        padding1 = np.ones((1, mesh_data.shape[1])) *(self.num_tokens +3)
        mesh_data = np.vstack([padding,mesh_data, padding1])  # Stack original data with padding
        if padding_needed > 0:
            # Pad with rows of zeros (or any other value you want)
            padding = np.ones((padding_needed, mesh_data.shape[1])) *(self.num_tokens +1) # pad with zeros, shape (padding_needed, 9)
            mesh_data = np.vstack([mesh_data, padding])  # Stack original data with padding
        if self.allmaxlen:
            padding = np.ones(( self.allmaxlen, mesh_data.shape[1])) *(1) # pad with zeros, shape (padding_needed, 9)
            mesh_data = np.vstack([mesh_data, padding])  # Stack original data with padding
        return mesh_data.reshape(-1).astype("int"),0
