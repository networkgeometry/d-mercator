import argparse
import textwrap
import numpy as np
import pandas as pd
import pyvista as pv
from matplotlib import cm
from tqdm import tqdm

pv.global_theme.color = 'white'


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent("""
    Visualize the S2 embeddings with communities.

    """))
    parser.add_argument('-i', '--embedding_path', type=str, required=True, help="Path to .inf_coord file")
    parser.add_argument('-l', '--label_path', type=str, required=True, help="""
        Path to file with nodes' labels. Format: 
        
        index0 label0
        index1 label1
        ...    ...     
    """)
    parser.add_argument('-e', '--edgelist', type=str, required=True, help="Path to .edge file")
    parser.add_argument('-r', '--resolution', type=int,
                        required=False, default=4096, help="Output image resolution.")
    parser.add_argument('-m', '--marker_scale', type=float,
                        required=False, default=0.09, help="Scale of marker.")
    
    parser.add_argument('-a', '--azimuth', type=float,
                        required=False, default=0, help="Camera azimuth. See more: https://docs.pyvista.org/api/core/camera.html")
    parser.add_argument('-v', '--elevation', type=float,
                        required=False, default=0, help="Camera elevation. See more: https://docs.pyvista.org/api/core/camera.html")

    parser.add_argument('-s', '--save_path', type=str, required=False, default='', help="Path to output PNG image.")
    args = parser.parse_args()
    return args


def get_geodesic(p1, p2):
    omega = np.arccos(np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2)))
    t = np.linspace(0, 1)
    
    line = []
    for t in np.linspace(0, 1):
        line.append(np.sin((1 - t) * omega) / np.sin(omega) * p1 + np.sin(t * omega) / np.sin(omega) * p2)
    return np.array(line)


def compute_prob_S2(beta, mu, p1, p2, kappa1, kappa2):
    R = 1
    angle = np.arccos(np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2)))
    chi = (R * angle) / np.sqrt(kappa1 * kappa2 * mu)
    return 1 / (1 + np.power(chi, beta))


def get_spherical_cap_structure_grid(b, opening_angle, R):
    # From: https://stackoverflow.com/a/45458451
    r = R
    phi = np.linspace(0, 2 * np.pi, 30)
    theta = np.linspace(0, opening_angle, 20)
    X = r * np.stack([
        np.outer(np.cos(phi), np.sin(theta)),
        np.outer(np.sin(phi), np.sin(theta)),
        np.outer(np.ones(np.size(phi)), np.cos(theta)),
        ], axis=-1)

    # rotate X such that [0, 0, 1] gets rotated to `c`;
    # <https://math.stackexchange.com/a/476311/36678>.
    a = np.array([0.0, 0.0, 1.0])
    a_x_b = np.cross(a, b)
    a_dot_b = np.dot(a, b)
    if a_dot_b == -1.0:
        X_rot = -X
    else:
        X_rot = (
            X +
            np.cross(a_x_b, X) +
            np.cross(a_x_b, np.cross(a_x_b, X)) / (1.0 + a_dot_b)
            )
        
    return pv.StructuredGrid(X_rot[..., 0], X_rot[..., 1], X_rot[..., 2])


def plot_embeddings(df, edges, beta, mu, save_path, resolution=4096, marker_scale=0.09, azimuth=0, elevation=0):
    pv.set_plot_theme("document")
    off_screen = True if save_path != '' else None
    plotter = pv.Plotter(off_screen=off_screen, window_size=[resolution, resolution])
    plotter.enable_anti_aliasing('ssaa')

    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    grid = pv.StructuredGrid(x, y, z)
    plotter.add_mesh(grid, color='#fdfdfd', opacity=1)

    # Plot edges
    pos = df[['p1', 'p2', 'p3']].values
    pos /= np.linalg.norm(pos, axis=1)[:, None]
    kappa = df['kappa'].values

    count = 0
    for source, target in tqdm(edges.values):
        s_i = df['index'].tolist().index(source)
        t_i = df['index'].tolist().index(target)
        
        # Compute the probability of connection
        p1, p2 = pos[s_i], pos[t_i]
        prob = compute_prob_S2(beta, mu, p1, p2, kappa[s_i], kappa[t_i])
        if prob < 0.5: # filter out low probable links
            count += 1
            continue

        l = get_geodesic(p1, p2)
        plotter.add_lines(l, color='#8a8a8a', width=6*prob)
      
    print('Number of low probable links: ', count)

    n_colors = len(np.unique(df['label']))
    colors = cm.tab10(np.linspace(0, 1, n_colors))
    
    idx = 0
    max_kappa = max(df['kappa'])
    for _, group in df.groupby("label"):
        pos = group[['p1', 'p2', 'p3']].values
        for j in range(len(group)):
            p = pos[j] / np.linalg.norm(pos[j])
            s = group['kappa'].values[j]
            s /= max_kappa
            s *= marker_scale
            cap = get_spherical_cap_structure_grid(p, s, 1.002)
            plotter.add_mesh(cap, color=colors[idx])
        idx += 1
    plotter.camera_position = 'yz'

    plotter.camera.azimuth = azimuth
    plotter.camera.elevation = elevation
        
    if save_path != '':
        plotter.screenshot(save_path)
    else:
        plotter.show()
    

def load_embeddings(path):
    df_S2 = pd.read_csv(path, sep="\s+", comment="#", header=None)
    df_S2.columns = ['index', 'kappa', 'hyp_radius', 'p1', 'p2', 'p3']
    pos = df_S2[['p1', 'p2', 'p3']].values
    df_S2[['p1', 'p2', 'p3']] = pos / np.linalg.norm(pos, axis=1)[:, None]
    return df_S2


def load_labels(path):    
    labels = pd.read_csv(path, sep="\s+", header=None)
    labels.columns = ['index', 'label']
    return labels


def load_edgelist(path):
    edges = pd.read_csv(path, sep="\s+", header=None)
    edges.columns = ['source', 'target']
    return edges


def extract_beta_mu(path):
    beta = -1
    mu = -1
    with open(path, 'r') as f:
        for line in f:
            if '- beta:' in line:
                beta = float(line.split(':')[-1])
            if '- mu: ' in line:
                mu = float(line.split(':')[-1])
    return beta, mu


if __name__ == '__main__':
    args = parse_args()

    df = load_embeddings(args.embedding_path)
    labels = load_labels(args.label_path)
    df = df.merge(labels, on='index')
    edges = load_edgelist(args.edgelist)

    beta, mu = extract_beta_mu(args.embedding_path)

    plot_embeddings(df, edges, beta, mu, args.save_path, args.resolution, 
                    args.marker_scale, args.azimuth, args.elevation)

