
from scipy.stats import mannwhitneyu, ttest_ind, pearsonr, linregress
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, PathPatch, FancyArrowPatch
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D
from matplotlib.cm import viridis, jet
import mpl_toolkits.mplot3d.art3d as art3d
from mpl_toolkits.mplot3d.proj3d import proj_transform

import numpy as np


def boxplot(data, test_type='mann-whitney', test_combinations=None, multiple_correction=None, custom_significant_combinations=None, show_points=False, show_connecting_lines=False, sep_multiplier=1, connecting_lines_skip=1, box_colors='w', box_widths=0.5, median_colors='k', boxes_alpha=1, linewdith=1, median_linewidth=1, points_colors='gradient', significance_lines_position='up'):

    # Se i dati sono una matrice di numpy, li trasformo in una lista di array
    if type(data) is np.ndarray:
        data = [data[:, i] for i in range(data.shape[1])]

    # Converto per sicurezza tutti gli elementi delle liste in array di numpy
    data = [np.array(d) for d in data]

    # Elimino eventuali nan
    data = [d[np.isnan(d) == False] for d in data]

    # Info sui dati
    boxes_num = len(data)
    samples_num = len(data[0])
    same_num = sum([True if len(data[i]) == len(data[0]) else False for i in range(boxes_num)]) == boxes_num

    # Mostro il boxplot usando la funzione di matplotlib
    bplot = plt.boxplot(data, patch_artist=True, widths=box_widths)

    # Coloro i boxplot e cambio le dimensioni delle linee
    for i, (patch, median) in enumerate(zip(bplot['boxes'], bplot['medians'])):

        if type(box_colors) is str:
            b_col = box_colors
        else:
            b_col = box_colors[i]

        if type(median_colors) is str:
            m_col = median_colors
        else:
            m_col = median_colors[i]

        patch.set_facecolor(b_col)
        median.set_color(m_col)
        patch.set_alpha(boxes_alpha)
        patch.set_linewidth(linewdith)
        median.set_linewidth(median_linewidth)

    # Mostro i punti
    if show_connecting_lines:
        show_points = True

    if show_points:

        if same_num:
            if type(points_colors) == str:
                if points_colors == 'random':
                    points_colors = [[np.random.rand(3) for _ in range(samples_num)]] * boxes_num

                elif points_colors == 'gradient':

                    # Ordino i sample dall'alto verso il basso nel primo box e uso quelli come riferimento
                    if connecting_lines_skip == 1:
                        y = np.array(data[0])
                        y_ind_sort = np.argsort(y)
                        points_colors = np.zeros((samples_num, 4))
                        for i in range(samples_num):
                            points_colors[y_ind_sort[i], :] = viridis(i/samples_num)
                        points_colors = [points_colors] * boxes_num
                    else:
                        points_colors = []
                        for i in range(0, boxes_num):
                            if i % connecting_lines_skip == 0:
                                y = np.array(data[i])
                                y_ind_sort = np.argsort(y)
                            p_col = np.zeros((samples_num, 4))

                            for j in range(samples_num):
                                p_col[y_ind_sort[j], :] = viridis(j/samples_num)
                            points_colors.append(p_col)

                elif points_colors == 'w':

                    points_colors = [np.ones((samples_num, 3))] * boxes_num

            elif type(points_colors) == np.ndarray:

                points_colors = [points_colors] * boxes_num

        else:
            points_colors = [[0, 0, 0]] * boxes_num
            print('Il tipo di colore richiesto per i punti è incompatibile con il fatto che ogni box ha un numero diverso di punti')

        for i in range(boxes_num-1):

            x_values_1 = np.ones(len(data[i])) * i + 1
            x_values_2 = np.ones(len(data[i+1])) * i + 2
            y_values_1 = data[i]
            y_values_2 = data[i+1]

            # Il colore dei punti o è dato da un gradiente (dall'alto in basso)
            # Oppure è dato dall'utente (per ogni box) oppure da un gradiente
            col_1 = points_colors[i]
            col_2 = points_colors[i+1]

            # Punti
            plt.scatter(x_values_1, y_values_1, 10, color=col_1, zorder=10, edgecolors='k')
            plt.scatter(x_values_2, y_values_2, 10, color=col_2, zorder=10, edgecolors='k')

    # Connetto tra loro i punti dei vari boxplot, ma solo per quelli adiacenti
    if show_connecting_lines:

        if same_num:

            for i in range(0, boxes_num-1, connecting_lines_skip):
                x_values_1 = np.ones(samples_num) * i + 1
                x_values_2 = np.ones(samples_num) * i + 2
                y_values_1 = data[i]
                y_values_2 = data[i+1]

                for j in range(samples_num):
                    if y_values_2[j] > y_values_1[j]:
                        line_col = 'g'
                    else:
                        line_col = 'r'

                    plt.plot([x_values_1[j], x_values_2[j]], [y_values_1[j], y_values_2[j]], line_col, alpha=0.2, linewidth=1)
        else:
            print("E' stato richiesto di mostrare le linee che connettono i punti nei box, ma ogni box ha un numero di punti diverso")

    # Creazione combinazioni per il test
    if test_combinations == 'all':
        test_combinations = [(i, j) for i in range(boxes_num) for j in range(i+1, boxes_num)]

    # Faccio i test
    if custom_significant_combinations is None:
        significant_combinations = []

        if test_combinations is not None:
            combinations_num = len(test_combinations)

            for c in test_combinations:

                data_1 = data[c[0]]
                data_2 = data[c[1]]

                if test_type == 't-test':
                    _, p = ttest_ind(data_1, data_2, alternative='two-sided')
                if test_type == 'mann-whitney':
                    _, p = mannwhitneyu(data_1, data_2, alternative='two-sided')

                if p <= 0.05:
                    significant_combinations.append((*c, p))

                print(c,p)
        # Correzione di Bonferroni
        if multiple_correction == 'bonferroni':
            for i, c in enumerate(significant_combinations):

                ind_1, ind_2, p = c
                p_mod = p * combinations_num

                if p_mod <= 0.05:
                    significant_combinations[i] = (ind_1, ind_2, p_mod)
                else:
                    significant_combinations.pop(i)
                    i -= 1

        # False discovery rate
        if multiple_correction == 'false-discovery':

            # Ordino la lista di tutti i p-value
            p_values = [c[2] for c in significant_combinations]
            p_sorted_inds = np.argsort(p_values)
            combinations_to_remove = []

            for i in range(len(p_sorted_inds)):
                p_ind = p_sorted_inds[i]
                ind_1, ind_2, p = significant_combinations[p_ind]
                p_mod = p * combinations_num / (i + 1)

                if p_mod <= 0.05:
                    significant_combinations[p_ind] = (ind_1, ind_2, p_mod)
                else:
                    combinations_to_remove.append(significant_combinations[p_ind])

            for c in combinations_to_remove:
                significant_combinations.remove(c)
    else:
        significant_combinations = custom_significant_combinations

    # Plotto le differenze significative
    if len(significant_combinations) > 0:

        print(significant_combinations)
        # Altezza iniziale per le linee
        ylim = plt.ylim()
        y_shift = 0.03 * (ylim[1] - ylim[0]) * sep_multiplier

        # Riordino le combinazioni in funzione della distanza tra i boxplot che le compongono
        box_dist = [abs(c[0] - c[1]) for c in significant_combinations]
        box_order = np.argsort(box_dist)
        significant_combinations_ordered = [significant_combinations[box_order[i]] for i in range(len(significant_combinations))]

        line_heights_up = np.array([np.max(d) for d in data])
        line_heights_down = np.array([np.min(d) for d in data])

        for i, c in enumerate(significant_combinations_ordered):
            ind1, ind2, p_value = c

            if p_value < 0.001:
                asterisks = '***'
            elif p_value < 0.01:
                asterisks = '**'
            elif p_value < 0.05:
                asterisks = '*'

            # Vedo l'altezza massima registrata di tutti i boxplot tra questi due
            # compresi essi stessi
            if significance_lines_position == 'up':
                line_pos = 'up'
            elif significance_lines_position == 'down':
                line_pos = 'down'
            elif type(significance_lines_position) is dict:
                # Cerco quella corrispondente
                if (ind1, ind2) in significance_lines_position:
                    line_pos = significance_lines_position[(ind1, ind2)]
                else:
                    line_pos = 'up'
            else:
                print('Tipo di posizionamento linee significatività non supportato')

            if line_pos == 'up':
                height = np.max(line_heights_up[ind1:(ind2+1)]) + y_shift
                tips_height = height
                line_height = tips_height + y_shift / 2
                line_heights_up[ind1:(ind2+1)] = height + y_shift * 2

            if line_pos == 'down':
                height = np.min(line_heights_down[ind1:(ind2+1)]) - y_shift
                tips_height = height
                line_height = tips_height - y_shift / 2
                line_heights_down[ind1:(ind2+1)] = height - y_shift * 2

            # Draw the significance line
            plt.plot([ind1 + 1, ind1 + 1, ind2 + 1, ind2 + 1], [tips_height, line_height, line_height, tips_height], lw=1, c='k')

            # Draw the asterisk for significance
            if line_pos == 'up':
                t_asterisk = plt.text((ind1 + ind2 + 2) * .5, line_height + y_shift * 0.3, asterisks, ha='center', va='bottom', color=[0.2, 0.2, 0.2])
            if line_pos == 'down':
                t_asterisk = plt.text((ind1 + ind2 + 2) * .5, line_height - y_shift * 0.3, asterisks, ha='center', va='top', color=[0.2, 0.2, 0.2])

            t_asterisk.set_bbox(dict(facecolor='white', alpha=1, linewidth=0, pad=0.05))


def plot_level_simple(level, rock_col=None, water_col=None):

    ax = plt.gca()

    if water_col is None:
        water_col = [204/255, 249/255, 255/255]
    ax.set_facecolor(water_col)

    for i, platform in enumerate(level.platforms):

        if rock_col is None:
            c = [0.7, 0.7, 0.7]
        else:
            c = rock_col
        edgecol = 'k'

        p = patches.Circle((platform.x, platform.y),
                           platform.size, edgecolor=edgecol, facecolor=c, label='_nolegend_')
        ax.add_patch(p)

def plot_Q_decision(level, Q, good_ids=None):

    cmap = viridis

    ax = plt.gca()

    for d in level.decision_points:
        p = d.platform
        id = p.id

        if good_ids is not None:
            if id not in good_ids: continue

        q = np.array(Q[id])
        q = np.exp(3 * q)
        q = q / np.sum(q)

        for j, n in enumerate(d.neighs):

            x = q[j]
            c = cmap(x)
            # print(x)

            circle = patches.Circle((n.x, n.y), n.size, edgecolor='k', facecolor=c, label='_nolegend_')
            ax.add_patch(circle)

            plt.arrow(p.x, p.y, (n.x - p.x)*0.5, (n.y - p.y)*0.5, color='r', width=0.1, head_width=0.6, zorder=10)

        # plt.plot([p.x], [p.y], 'X', color='w', markersize=8, markeredgecolor='k', markeredgewidth=1, zorder=20)


def plot_level(level, show_text=False, color_value=None, cmap=None):

    ax = plt.gca()

    bg_color = [204/255, 249/255, 255/255]
    platform_color_big = [176/255 * 0.9, 156/255 * 0.9, 141/255 * 0.9]
    platform_color_small = [201/255 * 0.9, 185/255 * 0.9, 173/255 * 0.9]
    platform_color_start = [0.4, 0.4, 0.4]

    ax.set_facecolor(bg_color)

    if color_value is not None:
        cmap = viridis
        color_value = (color_value - np.min(color_value))/(np.max(color_value) - np.min(color_value))

    for i, platform in enumerate(level.platforms):

        if platform.is_small:
            c = platform_color_small
        else:
            c = platform_color_big

        edgecol = 'None'

        if color_value is not None:
            c = cmap(color_value[i])


        p = patches.Circle((platform.x, platform.y),
                           platform.size, edgecolor=edgecol, facecolor=c, label='_nolegend_')
        ax.add_patch(p)

    platform_first = level.platforms[0]
    p = patches.Circle((platform_first.x, platform_first.y - 4),
                       2, edgecolor=None, facecolor=platform_color_start, label='_nolegend_')
    ax.add_patch(p)

    # Goal
    platform_last = level.platform_last
    p = patches.Circle((platform_last.x, platform_last.y + 4), 2,
                       linewidth=1, edgecolor=platform_color_start, facecolor=platform_color_start, label='_nolegend_')
    ax.add_patch(p)

    if show_text:
        plt.text(platform_last.x - 3, platform_last.y + 4, 'GOAL', fontsize=8, family='sans-serif', weight='bold', color=platform_color_start, ha='right', va='center')
        plt.text(platform_first.x + 3, platform_first.y - 4, 'START', fontsize=8, family='sans-serif', weight='bold', color=platform_color_start, ha='left', va='center')

    ax.relim()
    ax.autoscale_view()
    ax.set_aspect('equal')

    xlim = ax.get_xlim()
    xlim_l = xlim[1] - xlim[0]
    d = 50 - xlim_l

    plt.xlim([xlim[0] - d/2, xlim[1] + d/2])
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)


def plot_trajectory(trajectory):
    plt.plot(trajectory[:, 0], trajectory[:, 2])


def highlight_platform(level, platform_id):

    ax = plt.gca()
    platform = level.platforms[platform_id]
    p = patches.Circle((platform.x, platform.y), platform.size,
                       linewidth=1, edgecolor='r', facecolor='r', label='_nolegend_')
    ax.add_patch(p)


def plot_decision_points(level, predecision=False):

    for d in level.decision_points:
        p = d.platform

        for n in d.neighs:
            plt.arrow(p.x, p.y, (n.x - p.x)*0.6, (n.y - p.y)*0.6, color='r', width=0.1, head_width=0.6, zorder=10)

        plt.plot([p.x], [p.y], 'X', color='w', markersize=8, markeredgecolor='k', markeredgewidth=1, zorder=20)

    if predecision:
        for d in level.predecision_ids:
            p = level.platforms[d]
            plt.plot([p.x], [p.y], 'X', color='y', markeredgecolor='k', markeredgewidth=1, markersize=8, zorder=20)


def boxplot_lines(x, cols, sep=3, alpha=0.2, show_lines=False):

    # cols = [[1,0,0], [0,0,1]]
    cols_alpha = [[cols[0][0], cols[0][1], cols[0][2], alpha], [cols[1][0], cols[1][1], cols[1][2], alpha]]
    boxplot_significance_full(x, widths=0.3, correct_multiple=False, sep=sep)

    ylim = plt.ylim()

    if show_lines:
        for i in range(2):
            points_1 = x[:, i]
            points_2 = x[:, i+1]

            x_values_1 = np.ones(len(points_1)) * i - 0.25
            x_values_2 = np.ones(len(points_2)) * i + 0.25

            plt.plot(x_values_1, points_1, '.', color=cols[0], markeredgecolor='k')
            plt.plot(x_values_2, points_2, '.', color=cols[1], markeredgecolor='k')

            for j in range(len(points_1)):
                line_col = 'k'
                if points_2[j] > points_1[j]:
                    line_col = 'g'
                else:
                    line_col = 'r'

                plt.plot([x_values_1[j], x_values_2[j]], [points_1[j], points_2[j]], line_col, alpha=0.2, linewidth=1)

    plt.ylim(ylim)


def plot_adjacency_matrix(level, color='k', alpha=1):
    for i, p1 in enumerate(level.platforms):
        for j, p2 in enumerate(level.platforms):
            if j > i:
                if level.adjacency_matrix[i, j] == 1:
                    plt.plot([p1.x, p2.x], [p1.y, p2.y], '--', color=color, alpha=alpha, zorder=-1)


def plot_path(level, path, col='b', label='_nolegend_', alpha=1, linewidth=1):

    for i in range(len(path)):
        if i < len(path) - 1:
            p1 = level.platforms[path[i]]
            p2 = level.platforms[path[i+1]]

            if i == len(path) - 2:
                l = label
            else:
                l = '_nolegend_'

            plt.plot([p1.x, p2.x], [p1.y, p2.y], color='k', linewidth=linewidth * 2.5, zorder=90, label=l, alpha=alpha)
            plt.plot([p1.x, p2.x], [p1.y, p2.y], color=col, linewidth=linewidth, zorder=100, label=l, alpha=alpha)


def text3d(ax, xyz, s, zdir="z", size=None, angle=0, usetex=False, **kwargs):
    """
    Plots the string *s* on the axes *ax*, with position *xyz*, size *size*,
    and rotation angle *angle*. *zdir* gives the axis which is to be treated as
    the third dimension. *usetex* is a boolean indicating whether the string
    should be run through a LaTeX subprocess or not.  Any additional keyword
    arguments are forwarded to `.transform_path`.

    Note: zdir affects the interpretation of xyz.
    """
    x, y, z = xyz
    if zdir == "y":
        xy1, z1 = (x, z), y
    elif zdir == "x":
        xy1, z1 = (y, z), x
    else:
        xy1, z1 = (x, y), z

    text_path = TextPath((0, 0), s, size=size, usetex=usetex)
    trans = Affine2D().rotate(angle).translate(xy1[0], xy1[1])

    p1 = PathPatch(trans.transform_path(text_path), **kwargs)
    ax.add_patch(p1)
    art3d.pathpatch_2d_to_3d(p1, z=z1, zdir=zdir)


class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def visualize_trajectory_3d(trajectory, level, platform_values=None, cmap=None, decision_points=None):
    ax = plt.gca()

    # ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    ax.view_init(elev=40, azim=-90)
    ax.set_proj_type('ortho')
    ax.patch.set_alpha(0)
    if cmap is None:
        cmap = viridis

    if platform_values is not None:

        platform_values_list = list(platform_values.values())
        V_min = np.min(platform_values_list)
        V_max = np.max(platform_values_list)

    if decision_points is not None:
        decision_point_ids = [d.platform.id for d in level.decision_points]

    bg_color = [204/255, 249/255, 255/255]
    platform_color_big = [176/255, 156/255, 141/255]
    platform_color_small = [201/255, 185/255, 173/255]
    platform_color_start = [0.4, 0.4, 0.4]

    xmin = 1e10
    xmax = -1e10
    ymin = 1e10
    ymax = -1e10

    for platform in level.platforms:

        if platform.x < xmin:
            xmin = platform.x
        if platform.x > xmax:
            xmax = platform.x
        if platform.y < ymin:
            ymin = platform.y
        if platform.y > ymax:
            ymax = platform.y

    ymin -= 4
    ymax += 4

    ymin *= 1.1
    ymax *= 1.1

    x_width = 40
    x_center = (xmin + xmax)/2

    xmin = x_center - x_width/2
    xmax = x_center + x_width/2

    # Acqua
    p = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, facecolor=bg_color, zorder=-20)
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")

    # Piattaforme fiume
    for i, platform in enumerate(level.platforms):
        edgecol = 'None'

        if platform_values is not None:
            if i in platform_values:
                value_norm = (platform_values[i] - V_min) / (V_max - V_min)
                c = cmap(value_norm)
            else:
                c = 'k'

        elif decision_points is not None:
            if i in decision_point_ids:
                edgecol = 'r'

                if platform.is_small:
                    c = platform_color_small
                else:
                    c = platform_color_big
        else:
            if platform.is_small:
                c = platform_color_small
            else:
                c = platform_color_big

        p = patches.Circle((platform.x, platform.y),
                           platform.size, edgecolor=edgecol, linewidth=2, facecolor=c, label='_nolegend_', zorder=-10)
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")

    # # Decision points
    if decision_points is not None:
        for d in decision_points:
            for neigh in d.neighs:
                # arrow = patches.FancyArrowPatch((d.platform.x, d.platform.y), (neigh.x, neigh.y), color='r', arrowstyle='<|-', lw=3)
                # ax.add_patch(arrow)
                # art3d.pathpatch_2d_to_3d(arrow, z=0, zdir="z")
                delta_x = neigh.x - d.platform.x
                delta_y = neigh.y - d.platform.y
                arrow = Arrow3D(d.platform.x, d.platform.y, 0,
                                delta_x, delta_y, 0,
                                mutation_scale=15, arrowstyle="simple", zorder=150,
                                facecolor='r', ec=[0.5, 0, 0], alpha=0.8)
                ax.add_artist(arrow)

    # Prima piattaforma
    platform_first = level.platforms[0]
    p = patches.Circle((platform_first.x, platform_first.y - 4),
                       2, edgecolor=None, facecolor=platform_color_start, label='_nolegend_', zorder=-10)
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")

    text3d(ax, (platform_first.x + 3, platform_first.y - 4, 0), 'START', zdir="z", size=2, usetex=False,
           ec="none", fc=platform_color_start)

    # Ultima piattaforma
    platform_last = level.platform_last
    p = patches.Circle((platform_last.x, platform_last.y + 4),
                       2, edgecolor=None, facecolor=platform_color_start, label='_nolegend_', zorder=-10)
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")

    text3d(ax, (platform_last.x + 2, platform_last.y + 4, 0), 'GOAL', zdir="z", size=2, usetex=False,
           ec="none", fc=platform_color_start)

    # Traiettoria
    if type(trajectory) is list:
        for traj in trajectory:
            ax.plot(traj[:, 0], traj[:, 2], traj[:, 1]-0.2, alpha=0.5, zorder=10)

    else:
        ax.plot(trajectory[:, 0], trajectory[:, 2], trajectory[:, 1]-0.2, zorder=10, color='k')

    ax.set_xlim([xmin, xmax])
    # ax.set_ylim([-22, 15])
    ax.set_aspect('equal')

    plt.axis('off')


def get_data_for_condition(df, variable, conditions):

    result = []

    # Ciclo tra le condizioni
    for i, condition in enumerate(conditions):

        # Filtro i dati di questa condizione
        mask = None

        for var in condition:
            if mask is None:
                mask = df[var] == condition[var]
            else:
                mask = mask & (df[var] == condition[var])

        data_condition = df[mask]

        # Raggruppo per soggetto e medio su ogni variabile di interesse
        data_condition_by_subject = data_condition.groupby('SubjectID')

        var_mean = data_condition_by_subject[variable].mean().to_numpy()

        result.append(var_mean)

    return result


def boxplot_2x2(ax, df, variable, condition, significant_pairs, box_colors, median_colors, xlabels, only_top_lines=False):

    plt.sca(ax)

    # Estraggo i dati che mi interessano
    x_cond = get_data_for_condition(df, variable, condition)

    significance_line_position = {(0, 2): 'down', (1, 3): 'down'}
    if only_top_lines is True:
        significance_line_position = 'up'

    box_colors = box_colors*2
    median_colors = median_colors*2

    # Plot
    boxplot(x_cond, custom_significant_combinations=significant_pairs, boxes_alpha=0.8, box_colors=box_colors,
            median_colors=median_colors, significance_lines_position=significance_line_position, sep_multiplier=1)

    # Allargo un po' il range in modo da far respirare il plot
    y_lim = plt.ylim()
    y_range = (y_lim[1] - y_lim[0])
    space = y_range * 0.05
    y_lim_new = [y_lim[0] - space, y_lim[1] + space]

    # Disegno la linea verticale come separatore e
    plt.plot([2.5, 2.5], y_lim_new, ':k')
    plt.ylim(y_lim_new)

    plt.xticks([1.5, 3.5], xlabels, fontsize=15)
    plt.grid(True, linestyle=':')







def linear_fit_errors(x_mean, y_mean, x_std, y_std, samples=200):

    N = len(x_mean)
    slopes = np.zeros(samples)
    intercepts = np.zeros(samples)
    corrcoeff = np.zeros(samples)

    if x_std is None: x_std = 0
    if y_std is None: y_std = 0
    
    for n in range(samples):

        x = np.random.randn(N) * x_std + x_mean
        y = np.random.randn(N) * y_std + y_mean

        # coeff = np.polyfit(x,y,1)
        # m = coeff[0]
        # q = coeff[1]
        res = linregress(x,y)
        m = res.slope
        q = res.intercept
        r = res.rvalue

        slopes[n] = m
        intercepts[n] = q
        corrcoeff[n] = r

    return slopes, intercepts, corrcoeff

def linear_plot_errors(x_mean, y_mean, x_std=None, y_std=None, samples=1000, outliers_sigma=3, discard_outliers=False, show_outliers=True, col=[0, 0, 1]):

    mask_good = None
    outlier_inds = []

    if discard_outliers:
        if x_std is not None:
            x_std_std = np.std(x_std)
            mask_good = (np.abs((x_std - np.mean(x_std))/x_std_std) < outliers_sigma)
        if y_std is not None:
            y_std_std = np.std(y_std)

            if x_std is not None:
                mask_good = mask_good & (np.abs((y_std - np.mean(y_std))/y_std_std) < outliers_sigma)
            else:
                mask_good = (np.abs((y_std - np.mean(y_std))/y_std_std) < outliers_sigma)

    if mask_good is None:
        slopes, intercepts, corrcoeff = linear_fit_errors(x_mean, y_mean, x_std, y_std, samples=samples)
    else:
        x_std_good = None
        y_std_good = None
        outlier_inds = np.where(mask_good == False)[0]
        if x_std is not None:
            x_std_good = x_std[mask_good]
        if y_std is not None:
            y_std_good = y_std[mask_good]
        slopes, intercepts, corrcoeff = linear_fit_errors(x_mean[mask_good], y_mean[mask_good], x_std_good, y_std_good, samples=samples)

        print(f'Rimossi {len(x_mean) - np.sum(mask_good)} outliers')

    x_min = np.min(x_mean)
    x_max = np.max(x_mean)

    if x_std is not None:
        x_min = np.min(x_mean - x_std)
        x_max = np.max(x_mean + x_std)

    if mask_good is not None:
        x_min = np.min(x_mean[mask_good])
        x_max = np.max(x_mean[mask_good])

        if x_std is not None:
            x_min = np.min(x_mean[mask_good] - x_std[mask_good])
            x_max = np.max(x_mean[mask_good] + x_std[mask_good])

    deltax = x_max - x_min
    x_min -= deltax * 0.2
    x_max += deltax * 0.2
    x_interp = np.linspace(x_min, x_max, 30)

    y_interp_mean = np.zeros(len(x_interp))
    y_interp_min = np.zeros(len(x_interp))
    y_interp_max = np.zeros(len(x_interp))

    for i in range(len(x_interp)):
        x = x_interp[i]
        y_interp_mean[i] = np.mean(slopes) * x + np.mean(intercepts)

        y_distribution = np.zeros(samples)

        for n in range(samples):
            # slope = np.random.choice(slopes)
            # intercept = np.random.choice(intercepts)

            slope = slopes[n]
            intercept = intercepts[n]

            y = slope * x + intercept
            y_distribution[n] = y

        y_s = np.std(y_distribution)
        y_m = np.mean(y_distribution) - y_s
        y_M = np.mean(y_distribution) + y_s

        y_m = np.percentile(y_distribution, 5)
        y_M = np.percentile(y_distribution, 95)

        y_interp_min[i] = y_m
        y_interp_max[i] = y_M

    # Ora plotto
    # Dati con errore
    ecol = [col[0], col[1], col[2], 0.2]
    if show_outliers:
        plt.errorbar(x_mean, y_mean, xerr=x_std, yerr=y_std, marker='o', markersize=5, linestyle='none', capsize=3, ecolor=ecol)
    else:
        plt.errorbar(x_mean[mask_good], y_mean[mask_good], xerr=x_std[mask_good], yerr=y_std[mask_good], marker='o', markersize=5, linestyle='none', capsize=3, ecolor=ecol)

    # Linea
    plt.plot(x_interp, y_interp_mean, color=col)

    # Tutte le linee intermedie?
    # for n in range(samples):
    #     plt.plot(x_interp, x_interp * slopes[n] + intercepts[n], alpha=0.02, color='grey')

    # Area
    plt.fill_between(x_interp, y_interp_min, y_interp_max, alpha=0.2, color=col)

    # Outliers
    if show_outliers:
        for i in outlier_inds:
            plt.plot(x_mean[i], y_mean[i], '.', color='r', markersize=5, zorder=5)

    # Test
    if mask_good is None:
        corr = pearsonr(x_mean, y_mean)
    else:
        corr = pearsonr(x_mean[mask_good], y_mean[mask_good])

    # Faccio anche una regressione su tutto
    res = linregress(x_mean, y_mean)

    y_pred = np.mean(slopes) * x_mean + np.mean(intercepts)
    y_mean_mean = np.mean(y_mean)

    ss_res = np.sum((y_mean - y_pred)**2)
    ss_tot = np.sum((y_mean - y_mean_mean)**2)
    print(corr.statistic**2)
    print(1 - ss_res / ss_tot)
    plt.title(f'Pearson r = {corr.statistic:.2f}, p = {corr.pvalue:.4f}')




# \hline
# \hline
#  & \textbf{Coeff} & \textbf{Std Error} & \textbf{p-value} & \textbf{[0.025} & \textbf{0.975]} \\
# \hline
# Intercept & $-0.6781$ & $0.106$ & $\bm{< 0.001}$ & $-0.886$ & $-0.470$ \\
# Risky Jump Shorter Distance & $0.9396$ & $0.146$ & $\bm{< 0.001}$ & $0.653$ & $1.226$ \\
# Risky Jump Smaller Angle & $0.6384$ & $0.150$ & $\bm{< 0.001}$ & $0.344$ & $0.933$ \\
# Interaction & $0.7523$ & $0.227$ & $\bm{0.001}$ & $0.308$ & $1.197$ \\
# \hline
# \hline
# \end{tabular}
# \caption{Generalized Linear Model results for the probability of making the risky jump, showing the effects of distance and angle factors and their interaction.}
# \label{tab:FirstDecisionGLM}
# \end{table}