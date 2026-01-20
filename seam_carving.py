"""
Seam Carving (reduction de largeur)

Idée :
- On calcule une carte d’énergie (gradient)
- On modélise l’image comme un DAG :
    * noeud = pixel (r,c)
    * arcs vers la ligne suivante (r+1, c-1/c/c+1)
  => pas de cycles car on va uniquement de haut en bas.
- On trouve la couture (seam) d’énergie minimale = plus court chemin sur DAG
  (programmation dynamique en ordre topologique = ordre des lignes).
- On supprime la couture, on recommence k fois.

Hypothèse :
- l'image s'appelle : input.png
- elle est dans le même dossier que ce fichier
"""

import numpy as np
from PIL import Image
from dataclasses import dataclass


# ======================================================
# Chargement et sauvegarde de l'image
# ======================================================

def charger_image_rgb() -> np.ndarray:
    """Charge l'image input.png et la convertit en RGB."""
    return np.array(Image.open("input.png").convert("RGB"), dtype=np.uint8)


def sauvegarder_image_rgb(img: np.ndarray, nom: str) -> None:
    """Sauvegarde une image RGB sur le disque."""
    Image.fromarray(img.astype(np.uint8)).save(nom)


# ======================================================
# Calcul de la carte d'énergie
# ======================================================

def _rgb_vers_gris(img_rgb: np.ndarray) -> np.ndarray:
    """Conversion RGB -> niveaux de gris (formule standard)."""
    r = img_rgb[:, :, 0]
    g = img_rgb[:, :, 1]
    b = img_rgb[:, :, 2]
    return 0.299*r + 0.587*g + 0.114*b


def compute_energy_map(img_rgb: np.ndarray) -> np.ndarray:
    """
    Calcule la carte d'énergie de l'image.

    Comme vu en cours, l'énergie joue le rôle de coût associé aux sommets du DAG.
    """
    gray = _rgb_vers_gris(img_rgb)

    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)

    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]

    return np.abs(gx) + np.abs(gy)


# ======================================================
# Couture minimale = plus court chemin sur DAG
# ======================================================

@dataclass
class Seam:
    seam_cols: list
    total_energy: float


def find_min_vertical_seam(energy: np.ndarray) -> Seam:
    """
    Trouve la couture verticale minimale par programmation dynamique.

    Cette fonction implémente exactement le plus court chemin sur DAG
    en suivant l'ordre topologique naturel : ligne par ligne.
    """
    h, w = energy.shape
    dp = np.zeros((h, w))
    parent = np.zeros((h, w), dtype=int)

    # Initialisation : première ligne
    dp[0] = energy[0]

    # Relaxation en ordre topologique (lignes croissantes)
    for r in range(1, h):
        for c in range(w):
            candidats = [(dp[r-1, c], c)]
            if c > 0:
                candidats.append((dp[r-1, c-1], c-1))
            if c < w-1:
                candidats.append((dp[r-1, c+1], c+1))

            val, idx = min(candidats, key=lambda x: x[0])
            dp[r, c] = energy[r, c] + val
            parent[r, c] = idx

    # Fin du chemin
    c_min = int(np.argmin(dp[-1]))
    total = dp[-1, c_min]

    # Reconstruction du chemin
    seam_cols = [c_min]
    for r in range(h-1, 0, -1):
        c_min = parent[r, c_min]
        seam_cols.append(c_min)

    seam_cols.reverse()
    return Seam(seam_cols, total)


# ======================================================
# Suppression de la couture
# ======================================================

def remove_vertical_seam(img_rgb: np.ndarray, seam: Seam) -> np.ndarray:
    """
    Supprime la couture verticale de l'image.

    Chaque ligne perd exactement un pixel.
    """
    h, w, _ = img_rgb.shape
    new_img = np.zeros((h, w-1, 3), dtype=np.uint8)

    for r in range(h):
        c = seam.seam_cols[r]
        new_img[r] = np.delete(img_rgb[r], c, axis=0)

    return new_img


# ======================================================
# Pipeline principal
# ======================================================

def main() -> None:
    """
    Programme principal :
    - charge input.png
    - fixe k = moitié de la largeur
    - applique le seam carving k fois
    - sauvegarde uniquement l'image finale!!!
    """
    img = charger_image_rgb()
    h, w, _ = img.shape
    k = w // 2   # choix cohérent avec le sujet

    # Mettre True si on souhaite sauvegarder toutes les étapes intermédiaires
    save_steps = False

    if save_steps:
        sauvegarder_image_rgb(img, "step_000.png")

    for i in range(1, k + 1):
        energy = compute_energy_map(img)
        seam = find_min_vertical_seam(energy)
        img = remove_vertical_seam(img, seam)

        if save_steps:
            sauvegarder_image_rgb(img, f"step_{i:03d}.png")

    # Sauvegarde du résultat final
    sauvegarder_image_rgb(img, "image_reduite.png")
    print("Terminé : l'image finale est enregistrée sous le nom image_reduite.png dans le dossier du projet.")


if __name__ == "__main__":
    main()
