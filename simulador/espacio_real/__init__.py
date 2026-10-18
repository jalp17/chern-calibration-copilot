"""Chern en espacio real sobre red finita de Haldane (Tarea 1).

Modulos:
  hamiltonian_real_space - H(L, M, t1, t2, phi) 2L^2 x 2L^2
  chern_real_space       - Chern del subespacio ocupado (Bott index)
  dataset_real_space     - dataset (M, t1, t2, phi) -> etiqueta Chern

Uso minimo:
    from espacio_real import hamiltonian_real_space, chern_bott, dataset_real_space
    H = hamiltonian_real_space(L=3, M=1.0, t1=1.0, t2=0.20, phi=pi/2)
    C = chern_bott(H, n_occ=L**2)
"""

from hamiltonian_real_space import hamiltonian_real_space, lattice_vectors
from chern_real_space import chern_bott, chern_bott_batch, projector_occupied
from dataset_real_space import dataset_real_space

__all__ = [
    "hamiltonian_real_space",
    "lattice_vectors",
    "chern_bott",
    "chern_bott_batch",
    "projector_occupied",
    "dataset_real_space",
]