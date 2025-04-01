import numpy as np

def normalize(v):
    """Normalize a vector"""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.linalg.norm(np.array(p2) - np.array(p1))

def angle_between(v1, v2):
    """Calculate the angle between two vectors in radians"""
    v1_norm = normalize(v1)
    v2_norm = normalize(v2)
    dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    return np.arccos(dot)

def vector_to_angle(v):
    """Convert a vector to an angle in radians"""
    return np.arctan2(v[1], v[0])