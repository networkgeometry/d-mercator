from operator import itemgetter
from numba import jit
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats

COLORS = ['red', 'blue', 'green', 'orange', 'purple', 'black', 'pink']

@jit(nopython=True)
def euclidean_to_hyperspherical_coordinates(vec):
    # From: https://en.wikipedia.org/wiki/N-sphere
    # vec -- coordinates of node with size D+1
    r = np.linalg.norm(vec)
    angles = [r]
    for i in range(len(vec) - 2):
        bottom = 0
        for j in range(i, len(vec)):
            bottom += vec[j] * vec[j]
        bottom = np.sqrt(bottom)
        angles.append(np.arccos(vec[i] / bottom))

    denominator = np.sqrt(vec[-1] * vec[-1] + vec[-2] * vec[-2])
    if denominator < 1e-15:
        theta = 0
    else:
        theta = np.arccos(vec[-2] / denominator)
    
    if vec[-1] < 0:
        theta = 2 * np.pi - theta

    angles.append(theta)
    return angles


@jit(nopython=True)
def hyperspherical_to_euclidean_coordinates(v):
    positions = []
    angles = v[1:]
    r = v[0]
    for i in range(len(angles)):
        val = np.cos(angles[i])
        for j in range(i):
            val *= np.sin(angles[j])
        positions.append(r * val)

        if i == (len(angles) - 1):
            val = np.sin(angles[i])
            for j in range(i):
                val *= np.sin(angles[j])
            positions.append(r * val)
    return positions


@jit(nopython=True)
def compute_angular_distances(x, y):
    angular_distances = []
    for v, u in zip(x, y):
        angular_distances.append(
            np.arccos(np.dot(v, u) / (np.linalg.norm(v) * np.linalg.norm(u))))
    return angular_distances


def get_rotation_matrix_SD(u, v=np.matrix([1, 0, 0, 0], dtype=float)):
    # From https://math.stackexchange.com/a/4167838
    u = np.matrix(u)
    v = np.matrix(v)    
    uv = u + v
    
    R = np.eye(4) - uv.T / (1 + u @ v.T) * uv + 2 * np.multiply(u, v.T)
    return R


def rotation_matrix_XY(theta):
    m = np.matrix([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, np.cos(theta), np.sin(theta)],
                   [0, 0, -np.sin(theta), np.cos(theta)]])
    return m


def rotation_matrix_XW(theta):
    m = np.matrix([[1, 0, 0, 0],
                   [0, np.cos(theta), -np.sin(theta), 0],
                   [0, np.sin(theta), np.cos(theta), 0],
                   [0, 0, 0, 1]])
    return m


def apply_pipeline_matrix_S3(pos1, pos2, theta_num=50):
    mean_distances = []
    pos1_rotation_matrices = []
    pos2_rotation_matrices = []
    thetas1 = []
    thetas2 = []

    for i in tqdm(range(len(pos1))):
        node_i = pos1[i]
        node_i_prime = pos2[i]

        m_i = get_rotation_matrix_SD(node_i)
        m_j = get_rotation_matrix_SD(node_i_prime)

        pos1_axis = np.array([(m_i @ v).A1 for v in pos1])
        pos2_axis = np.array([(m_j @ v).A1 for v in pos2])

        for theta1 in np.linspace(0, 2*np.pi, num=theta_num):
            pos2_XY = np.matmul(rotation_matrix_XY(theta1), pos2_axis.transpose()).T

            for theta2 in np.linspace(0, 2*np.pi, num=theta_num):
                pos2_XY_XW = np.matmul(rotation_matrix_XW(theta2), pos2_XY.transpose()).T
                
                mean_distance = compute_angular_distances(
                    pos1_axis, pos2_XY_XW)
                mean_distances.append(mean_distance)

                pos1_rotation_matrices.append(m_i)
                pos2_rotation_matrices.append(m_j)
                thetas1.append(theta1)
                thetas2.append(theta2)

    results = pd.DataFrame(mean_distances).T
    min_distance_idx = np.argmin(results.mean(axis=0).values)
    min_pos1_rotation_matrix = pos1_rotation_matrices[min_distance_idx]
    min_pos2_rotation_matrix = pos2_rotation_matrices[min_distance_idx]
    min_theta1 = thetas1[min_distance_idx]
    min_theta2 = thetas2[min_distance_idx]

    out_pos1 = np.array([(min_pos1_rotation_matrix @ v).A1 for v in pos1])
    out_pos2 = np.array([(min_pos2_rotation_matrix @ v).A1 for v in pos2])
    out_pos2 = np.matmul(rotation_matrix_XY(min_theta1), out_pos2.transpose()).T
    out_pos2 = np.matmul(rotation_matrix_XW(min_theta2), out_pos2.transpose()).T
    out_pos2 = np.array(out_pos2)
    
    out_pos1_spherical = np.array([euclidean_to_hyperspherical_coordinates(v) for v in out_pos1])
    out_pos2_spherical = np.array([euclidean_to_hyperspherical_coordinates(v) for v in out_pos2])
    return out_pos1, out_pos2, out_pos1_spherical, out_pos2_spherical


def apply_pipeline_matrix_with_loading_S3(real_coords_path, inf_coords_path, cutoff_percent=0.005, theta_num=20, new_version=False):
    # Load nodes' positions
    real_coords = pd.read_csv(
        real_coords_path, sep="\s+", comment="#", header=None)
    if new_version:
        real_coords.columns = ['index', 'kappa', 'radius', 'pos0',
                            'pos1', 'pos2', 'pos3', 'realdeg', 'expdegree']
    else:
        real_coords.columns = ['index', 'kappa', 'pos0',
                            'pos1', 'pos2', 'pos3', 'realdeg', 'expdegree']
    inf_coords = pd.read_csv(
        inf_coords_path, comment="#", header=None, sep="\s+")
    if new_version:
        inf_coords.columns = ['index', 'inf_kappa', 'inf_hyp_radius',
                            'inf_pos0', 'inf_pos1', 'inf_pos2', 'inf_pos3']
    else:        
        inf_coords.columns = ['index', 'inf_kappa',
                            'inf_pos0', 'inf_pos1', 'inf_pos2', 'inf_pos3']
    df = inf_coords.merge(real_coords, on="index")
    # Filter nodes with lower than `min_degree` degree
    cutoff_degree = np.max(df["realdeg"]) * cutoff_percent
    print('Cutoff degree: ', cutoff_degree)
    df = df[df["realdeg"] >= cutoff_degree]
    real_coords_all = df[["pos0", "pos1", "pos2", "pos3"]].values
    inf_coords_all = df[["inf_pos0", "inf_pos1", "inf_pos2", "inf_pos3"]].values

    r1 = np.mean([np.linalg.norm(r) for r in real_coords_all])
    r2 = np.mean([np.linalg.norm(r) for r in inf_coords_all])
    real_coords_all /= r1
    inf_coords_all /= r2
    return apply_pipeline_matrix_S3(real_coords_all, inf_coords_all, theta_num=theta_num)


def apply_pipeline_matrix_with_loading_and_communities_S3(real_coords_path, inf_coords_path, label_path, cutoff_percent=0.005, theta_num=20):
    # Load nodes' positions
    real_coords = pd.read_csv(
        real_coords_path, sep="\s+", comment="#", header=None)
    real_coords.columns = ['index', 'kappa', 'pos0',
                           'pos1', 'pos2', 'realdeg', 'expdegree']
    inf_coords = pd.read_csv(
        inf_coords_path, comment="#", header=None, sep="\s+")
    inf_coords.columns = ['index', 'inf_kappa',
                          'inf_pos0', 'inf_pos1', 'inf_pos2']
    df = inf_coords.merge(real_coords, on="index")

    label_df = pd.read_csv(label_path, sep="\t", header=None)
    label_df.columns = ['pos0', 'pos1', 'pos2', 'label']
    label_values = label_df['label'].values

    # Filter nodes with lower than `min_degree` degree
    cutoff_degree = np.max(df["realdeg"]) * cutoff_percent
    print('Cutoff degree: ', cutoff_degree)
    df = df[df["realdeg"] >= cutoff_degree]
    real_coords_all = df[["pos0", "pos1", "pos2"]].values
    inf_coords_all = df[["inf_pos0", "inf_pos1", "inf_pos2"]].values

    label_idx = [int(s[1:]) for s in df['index']]
    for i in range(len(label_idx)):
        val = label_idx[i]
        if val >= len(label_values):
            label_idx[i] = len(label_values) - 1
        
    label_values = label_values[label_idx]

    return apply_pipeline_matrix_S3(real_coords_all, inf_coords_all, theta_num=theta_num), label_values


###################### PLOTTING #############################
def plot_distance_per_node_S3(results, title=''):
    plt.figure(figsize=(12, 5))
    x = list(range(len(results)))
    y = results.mean(axis=0).values
    yerr = results.std(axis=1).values
    y, yerr = [list(x) for x in zip(*sorted(zip(y, yerr), key=itemgetter(0)))]
    plt.errorbar(x, y, yerr=yerr, marker='.', alpha=0.6, elinewidth=0.3)
    plt.xlabel("Sorted Node index")
    plt.ylabel("Mean angular distance after rotation")
    plt.title(title)


def plot_spherical_coordinates_comparison_S3(best_inf_coords_spherical, real_coords_spherical, title='', labels=None):
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 3, 1)
    y = best_inf_coords_spherical[:, 1]
    x = real_coords_spherical[:, 1]

    if labels is not None:
        plt.scatter(x, y, marker='.', color=[COLORS[i] for i in labels], alpha=0.8)    
    else:
        plt.scatter(x, y, marker='.', alpha=0.8)
    xx = np.linspace(min(np.min(x), np.min(y)), max(np.max(x), np.max(y)))
    plt.plot(xx, xx, '--', c='k')

    res = stats.pearsonr(x, y)
    plt.title('pearson={:.3f}'.format(res[0]))
    plt.xlabel(r'$\varphi^{\textrm{real}}_1$')
    plt.ylabel(r'$\varphi^{\textrm{inferred}}_1$')

    plt.subplot(1, 3, 2)
    y = best_inf_coords_spherical[:, 2]
    x = real_coords_spherical[:, 2]
    if labels is not None:
        plt.scatter(x, y, marker='.', color=[COLORS[i] for i in labels], alpha=0.8)    
    else:
        plt.scatter(x, y, marker='.', alpha=0.8)
    
    xx = np.linspace(min(np.min(x), np.min(y)), max(np.max(x), np.max(y)))
    plt.plot(xx, xx, '--', c='k')

    res = stats.pearsonr(x, y)
    plt.title('pearson={:.3f}'.format(res[0]))
    plt.xlabel(r'$\varphi^{\textrm{real}}_2$')
    plt.ylabel(r'$\varphi^{\textrm{inferred}}_2$')
    
    plt.subplot(1, 3, 3)
    y = best_inf_coords_spherical[:, 3]
    x = real_coords_spherical[:, 3]
    if labels is not None:
        plt.scatter(x, y, marker='.', color=[COLORS[i] for i in labels], alpha=0.8)    
    else:
        plt.scatter(x, y, marker='.', alpha=0.8)
    
    xx = np.linspace(min(np.min(x), np.min(y)), max(np.max(x), np.max(y)))
    plt.plot(xx, xx, '--', c='k')

    res = stats.pearsonr(x, y)
    plt.title('pearson={:.3f}'.format(res[0]))
    plt.xlabel(r'$\varphi^{\textrm{real}}_3$')
    plt.ylabel(r'$\varphi^{\textrm{inferred}}_3$')

    plt.suptitle(title, fontsize=26, y=1.02)



def plot_euclidean_coordinates_comparison_S3(best_inf_coords_euclidean, real_coords_all, title='', labels=None):
    plt.figure(figsize=(18, 5))
    plt.subplots_adjust(wspace=0.3)
    plt.subplot(1, 4, 1)
    y = best_inf_coords_euclidean[:, 0]
    x = real_coords_all[:, 0]

    if labels is not None:
        plt.scatter(x, y, marker='.', color=[COLORS[i] for i in labels], alpha=0.8)    
    else:
        plt.scatter(x, y, marker='.', alpha=0.8)
    xx = np.linspace(min(np.min(x), np.min(y)), max(np.max(x), np.max(y)))
    plt.plot(xx, xx, '--', c='k')

    res = stats.pearsonr(x, y)
    plt.title('pearson={:.3f}'.format(res[0]))
    plt.xlabel(r'$X^{\textrm{real}}_1$')
    plt.ylabel(r'$X^{\textrm{inferred}}_1$')

    plt.subplot(1, 4, 2)
    y = best_inf_coords_euclidean[:, 1]
    x = real_coords_all[:, 1]
    if labels is not None:
        plt.scatter(x, y, marker='.', color=[COLORS[i] for i in labels], alpha=0.8)    
    else:
        plt.scatter(x, y, marker='.', alpha=0.8)
    
    xx = np.linspace(min(np.min(x), np.min(y)), max(np.max(x), np.max(y)))
    plt.plot(xx, xx, '--', c='k')

    res = stats.pearsonr(x, y)
    plt.title('pearson={:.3f}'.format(res[0]))

    plt.xlabel(r'$X^{\textrm{real}}_2$')
    plt.ylabel(r'$X^{\textrm{inferred}}_2$')

    plt.subplot(1, 4, 3)
    y = best_inf_coords_euclidean[:, 2]
    x = real_coords_all[:, 2]
    if labels is not None:
        plt.scatter(x, y, marker='.', color=[COLORS[i] for i in labels], alpha=0.8)    
    else:
        plt.scatter(x, y, marker='.', alpha=0.8)
    
    xx = np.linspace(min(np.min(x), np.min(y)), max(np.max(x), np.max(y)))
    plt.plot(xx, xx, '--', c='k')

    res = stats.pearsonr(x, y)
    plt.title('pearson={:.3f}'.format(res[0]))

    plt.xlabel(r'$X^{\textrm{real}}_3$')
    plt.ylabel(r'$X^{\textrm{inferred}}_3$')
    
    plt.subplot(1, 4, 4)
    y = best_inf_coords_euclidean[:, 3]
    x = real_coords_all[:, 3]
    if labels is not None:
        plt.scatter(x, y, marker='.', color=[COLORS[i] for i in labels], alpha=0.8)    
    else:
        plt.scatter(x, y, marker='.', alpha=0.8)
    
    xx = np.linspace(min(np.min(x), np.min(y)), max(np.max(x), np.max(y)))
    plt.plot(xx, xx, '--', c='k')

    res = stats.pearsonr(x, y)
    plt.title('pearson={:.3f}'.format(res[0]))

    plt.xlabel(r'$X^{\textrm{real}}_4$')
    plt.ylabel(r'$X^{\textrm{inferred}}_4$')

    plt.suptitle(title, fontsize=26, y=1.02)
