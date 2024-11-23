import torch
import theseus as th

from torch import Tensor
from theseus.geometry import SO3
from typing import Callable

def is_tan(t):
  if isinstance(t, Tensor) and t.shape[-2:] == (1, 3):
    return True
  else:
    return False
  
def is_mat(t):
  if isinstance(t, Tensor) and t.shape[-2:] == (3, 3):
    return True
  else:
    return False

def as_tan(t)-> Tensor:
  if isinstance(t, SO3):
    return t.log_map()
  if is_tan(t):
    return t
  raise ValueError(t.shape)

def as_lie(t) -> SO3:
  if isinstance(t, SO3):
    return t
  if is_tan(t):
    # from tangent vector
    return SO3.exp_map(t)
  if is_mat(t):
    # from 3x3 matrix
    return SO3(tensor = t)
  raise ValueError(t.shape)

def as_mat(t)-> Tensor:
  if isinstance(t, SO3):
    return t.to_matrix()
  if is_mat(t):
    return t
  raise ValueError(t.shape)

def as_repr(t, repr: str):
  if repr == "lie":
    return as_lie(t)
  elif repr == "tan":
    return as_tan(t)
  elif repr == "mat":
    return as_mat(t)
  raise ValueError(repr)


def chordal_distance(x1, x2):
  m1 = as_mat(x1)
  m2 = as_mat(x2)
  m = (m1 - m2) ** 2
  m = m.reshape((*m.shape[:-2], m.shape[-2] * m.shape[-1]))
  return m.sum(axis=-1)

distance_dict = {
  # "geodesic": geodesic_distance,
  "chordal": chordal_distance,
  # "quaternion": quaternion_distance,
  # "hyperbolic": hyperbolic_distance,
  # "chordal6d": chordal6d_distance,
}


def get_distance_fn(type="chordal") -> Callable:
  type = str(type).lower()
  assert type in distance_dict.keys()

  distance_fn = distance_dict[type]
  return distance_fn


def main():
  matrix = SO3.rand(1)
  matrix2 = SO3.rand(2)
  # print(matrix)
  vector = as_tan(matrix)
  # print(vector)
  m = as_mat(matrix)
  m = as_lie(m)
  # print(m)
  x = chordal_distance(matrix, matrix2)
  print(x)


if __name__ == "__main__":
    main()