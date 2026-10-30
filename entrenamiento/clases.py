"""Clases del modelo de suelo de Supergiz (compartidas por preparación, entrenamiento y nodo)."""

IGNORAR = 255

CLASES = [
    'otro',          # 0: edificios, coches, vegetación... (no es suelo transitable)
    'acera',         # 1
    'calzada',       # 2
    'paso_cebra',    # 3
    'bordillo',      # 4
    'escalera',      # 5
    'agujero',       # 6
]

# Barcelona Streets (Roboflow, bielglasses) -> nuestras clases
BARCELONA = {
    'sidewalk': 1, 'floor': 1, 'ground': 1,
    'road': 2,
    'crosswalk': 3,
    'curb-up': 4, 'curb-down': 4,
    'stairs-up': 5, 'stairs-down': 5,
    'hole': 6,
}

# Orden de pintado: lo fino y lo peligroso encima de lo grande
PRIORIDAD = [2, 1, 3, 4, 5, 6]

TRANSITABLES = ['acera', 'paso_cebra']
