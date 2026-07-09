"""
Palette de couleurs partagée pour représenter les "types" d'anomalies.

Okabe-Ito : conçue pour rester distinguable même en daltonisme (proto/
deutéranopie, tritanopie) — contrairement à des teintes choisies à l'œil
(ex. cyan/turquoise ou orange/orange foncé) qui se confondent, ou à un
colormap continu (ex. Set2) rééchantillonné sur peu de types, qui produit
des teintes trop proches.
"""

TYPE_COLORS = [
    '#e69f00',  # 0 orange
    '#56b4e9',  # 1 bleu ciel
    '#009e73',  # 2 vert
    '#f0e442',  # 3 jaune
    '#0072b2',  # 4 bleu
    '#d55e00',  # 5 vermillon
    '#cc79a7',  # 6 violet rosé
    '#000000',  # 7 noir
]


def shared_pair_color(i):
    """
    Couleur attribuée à la i-ème paire de types partagés
    (oscillo_correlation.find_shared_types). Chaque paire a sa propre
    couleur (les deux côtés de LA MÊME paire partagent cette couleur), sans
    essayer de garder une couleur cohérente entre paires différentes : un
    type très corrélé avec plusieurs autres (un "hub") fusionnerait sinon
    tout le monde en une seule couleur par transitivité, ce qui rend la
    section illisible.
    """
    return TYPE_COLORS[i % len(TYPE_COLORS)]


def build_color_map(pairs):
    """
    Construit un mapping {(signal_key, cluster_id): couleur_hex} stable à
    partir d'une liste ordonnée de paires (signal, type) déjà dédupliquées.
    Chaque paire reçoit UNE couleur nettement distincte des autres — un
    type = une couleur, peu importe le signal.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # Palette à couleurs franchement distinctes (pas de resampling continu d'un
    # colormap, qui produit des teintes quasi identiques quand peu de paires) :
    # on réutilise TYPE_COLORS, puis on complète avec tab20 (discret) si besoin.
    palette = list(TYPE_COLORS)
    if len(pairs) > len(palette):
        extra_cmap = plt.cm.get_cmap('tab20')
        palette += [mcolors.to_hex(extra_cmap(i)) for i in range(20)]

    return {pair: palette[i % len(palette)] for i, pair in enumerate(pairs)}
