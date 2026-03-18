import json
import numpy as np
import pandas as pd
from typing import List


class Platform():

    def __init__(self, x, y, s, id):
        self.x = x
        self.y = y
        self.id = id
        self.size = s

        if self.size < 0.9:
            self.is_small = True
        else:
            self.is_small = False


class DecisionPoint():

    def __init__(self, platform, neighs, paths):

        self.platform = platform
        self.neighs = neighs
        self.paths = paths


class Level():

    def __init__(self, filename):

        f = open(f'Levels/{filename}.json')
        data = json.load(f)

        self.level_name = filename

        # Dal nome del livello deduco tutte le informazioni
        self.is_flipped = False
        self.is_training = False
        self.shortcut = False

        c = self.level_name.split('_')
        if len(c) == 1:
            self.is_training = True
        else:
            if c[1] == 's':
                self.shortcut = True

            if len(c) == 3:
                self.is_flipped = True

        self.level_id = c[0]

        # Creo tutte le piattaforme
        self.platforms = []

        platform_info = data['platformInfo']
        for i, p in enumerate(platform_info):

            x = p['position']['x']
            y = p['position']['y']
            s = p['scale']

            platform = Platform(x, y, s, i)
            self.platforms.append(platform)

    def create_graph(self):

        self.adjacency_matrix = np.zeros(
            (len(self.platforms), len(self.platforms)), dtype=np.int32)
        for i, p1 in enumerate(self.platforms):
            for j, p2 in enumerate(self.platforms):
                if i != j:
                    if np.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2) < 4.7:
                        self.adjacency_matrix[i, j] = 1

    def find_last_platform(self):

        platform_last = None
        platform_last_y = -1e10

        for i, platform in enumerate(self.platforms):

            if platform.y > platform_last_y:
                platform_last_y = platform.y
                platform_last = i

        self.platform_last = self.platforms[platform_last]

    def find_decision_points(self):
        # A partire dalla matrice di adiacenza cerco i punti che hanno almeno
        # due vicini (entrambi più vicini al goal lungo Y). Inoltre considero 
        # solo i casi in cui una delle due pietre su cui salto è piccola
        decision_points = []
        old_neighs = []

        for i in range(len(self.platforms)):
            neigh_num = 0
            neighs = []
            one_small = False

            # Controllo che questa pietra non sia uno dei vicini precedenti
            # if self.platforms[i].id in old_neighs: continue

            # Per ora considero solo pietre grandi
            if self.platforms[i].is_small:
                continue

            for j in range(len(self.platforms)):
                if self.adjacency_matrix[j, i] == 1:
                    if self.platforms[j].y > self.platforms[i].y:
                        neigh_num += 1
                        neighs.append(self.platforms[j])

                        if self.platforms[j].is_small:
                            one_small = True


            if neigh_num == 2 and one_small:

                # Ora che l'ho trovato devo anche calcolare tutti i path
                # che vanno al goal passando dai vicini
                paths = []
                for k, n in enumerate(neighs):
                    # old_neighs.append(n.id)

                    if k == 0: other_id = neighs[1].id
                    if k == 1: other_id = neighs[0].id
                    path = self.dijkstra(n.id, omit=other_id)

                    # path = self.dijkstra(n.id)
                    paths.append(path)

                decision_point = DecisionPoint(platform=self.platforms[i], neighs=neighs, paths=paths)
                decision_points.append(decision_point)
            
        self.decision_points = decision_points

    def find_predecision_points(self):

        decision_ids = []
        predecision_ids = []

        for i in range(len(self.decision_points)):
            decision_ids.append(self.decision_points[i].platform.id)

        for i in range(len(self.decision_points)):
            decision_id = self.decision_points[i].platform.id
            
            # Vedo tutti i vicini
            for j in range(len(self.platforms)):
                if self.adjacency_matrix[j, decision_id] == 1:
                    if self.platforms[j].y < self.platforms[decision_id].y:
                        if j not in decision_ids:
                            predecision_ids.append(j)

        self.decision_ids = decision_ids
        self.predecision_ids = predecision_ids

        # Trovo anche i punti di "non decisione" e li considero
        # come quelli che non sono né di un tipo né dell'altro
        # ma che al contempo hanno solo un vicino
        all_platform_nondecision_ids = [i for i in range(len(self.platforms)) if i not in decision_ids and i not in predecision_ids]
        nondecision_proper_ids = []

        for i in all_platform_nondecision_ids:

            # Controllo che abbia un solo vicino sopra
            neigh_up_count = 0
            for j in range(len(self.platforms)):
                if self.adjacency_matrix[i,j] == 1:
                    if self.platforms[j].y > self.platforms[i].y:
                        neigh_up_count += 1
            if neigh_up_count == 1:
                nondecision_proper_ids.append(i)
        
        self.nondecision_ids = nondecision_proper_ids
        


    def dijkstra(self, source=0, target=None, omit=None):

        # Identifico il target se non specificato
        if target is None:
            target = self.platform_last.id

        # Copio la matrice d'adiacenza poiché potrei modificarla
        A = np.copy(self.adjacency_matrix)

        # Elimino tutti i nodi omessi
        if omit is not None:
            A[omit, :] = 0
            A[:, omit] = 0

        # Elimino tutto i nodi più in basso del source
        for i in range(len(self.platforms)):
            if self.platforms[i].y < self.platforms[source].y:
                A[i, :] = 0
                A[:, i] = 0
            else:
                break

        dist = np.zeros((len(self.platforms)), dtype=np.float32)
        prev = np.zeros((len(self.platforms)))

        Q = []
        for i in range(len(self.platforms)):
            dist[i] = 1e100
            prev[i] = None
            Q.append(i)

        dist[source] = 0

        while len(Q) > 0:

            # Trovo il vertice con distanza minore
            min_dist = 1e100
            min_ind = -1

            for i in range(len(Q)):
                ind = Q[i]
                d = dist[ind]

                if d < min_dist:
                    min_dist = d
                    min_ind = ind

            if min_ind == target:
                break

            Q.remove(min_ind)

            # Ciclo tra i vicini di questo vertice
            A_i = A[min_ind, :]

            for i in range(len(A_i)):
                if A_i[i] == 1:
                    if i in Q:
                        alt = min_dist + 1
                        if alt < dist[i]:
                            dist[i] = alt
                            prev[i] = min_ind

        S = []
        u = target

        if prev[u] is not None or u == 0:
            while u is not None and not np.isnan(u):
                S.insert(0, int(u))
                u = prev[int(u)]

        return S


class Player():

    def __init__(self, data, prolific_ID, levels):

        self.level_results = {}
        self.prolific_ID = prolific_ID

        fps = 0
        count = 0

        for key in data:
            if key == 'Info':
                self.age = int(data['Info']['age'])
                self.gender = data['Info']['gender']
            else:
                if key != 'FinalResults':
                    level_name = '_'.join(key.split('_')[2:])

                    if not level_name.startswith('Training'):
                        level_data = data[key]
                        fps += float(level_data['averageFPS'])
                        count += 1

                        # Trovo il livello corrispondente
                        for level in levels:
                            if level.level_name == level_name:
                                break

                        self.level_results[level_name] = PlayerLevelResult(
                            level_data, level)
                else:
                    self.total_score = data['FinalResults']['score']
                    self.total_time = data['FinalResults']['totalTime']

        if count > 0:
            self.fps = fps / count


class Event():
    def __init__(self, t, x, y, type):
        self.time = t
        self.x = x
        self.y = y

        if type == 0:
            self.type = 'Platform'

        elif type == 1:
            self.type = 'Won'

        elif type == 2:
            self.type = 'Water'

        elif type == 4:
            self.type = 'LostTime'
        
        else:
            print(type)


class PlayerLevelResult():

    def __init__(self, data, level):

        data_trajectory = data['data']
        data_events = data['events']

        N = len(data_trajectory)

        self.timepoints = np.zeros((N, 1))
        self.trajectory = np.zeros((N, 3))
        self.angles = np.zeros((N, 1))
        self.events = []

        for i, p in enumerate(data_trajectory):
            x = p.split(';')
            self.timepoints[i] = float(x[0])
            self.trajectory[i, :] = [float(x[1]), float(x[2]), float(x[3])]
            self.angles[i] = float(x[4])

        for e in data_events:
            x = e.split(';')
            event = Event(float(x[0]), float(x[1]), float(x[2]), int(x[3]))
            self.events.append(event)

        # Ciclo su tutti gli eventi di tipo "piattaforma"
        last_platform_id = -1
        trajectory_platforms = []
        time_platforms = []
        first_platform_time = -1

        for e in self.events:
            if e.type == 'Platform':
                
                # Trovo la piattaforma reale più vicina
                min_dist = 1e10
                min_ind = -1

                for k, p in enumerate(level.platforms):
                    dist = np.sqrt((p.x - e.x)**2 + (p.y - e.y)**2)
                    if dist < min_dist:
                        min_dist = dist
                        min_ind = k

                if first_platform_time == -1:
                    first_platform_time = e.time

                if min_ind != last_platform_id:
                    last_platform_id = min_ind
                    trajectory_platforms.append(min_ind)
                    time_platforms.append(e.time - first_platform_time)

        self.trajectory_platforms = trajectory_platforms
        self.time_platforms = time_platforms


class DecisionPointData():

    def __init__(self, decision_point, time, neigh_small, distance, euclidean_distance,
                 angle_goal, angle_trajectory, level_name, player_id,
                 path, trajectory_platforms, trajectory_real, distance_diff,
                 angle_goal_diff, angle_trajectory_diff, 
                 normalized_trial_time, unique_rock_id):
        
        self.decision_point = decision_point
        self.time = time
        self.neigh_small = neigh_small
        self.distance = distance
        self.euclidean_distance = euclidean_distance
        self.angle_goal = angle_goal
        self.angle_trajectory = angle_trajectory
        self.level_name = level_name
        self.player_id = player_id
        self.best_path = path
        self.trajectory_platforms = trajectory_platforms
        self.trajectory_real = trajectory_real
        self.distance_diff = distance_diff
        self.angle_goal_diff = angle_goal_diff
        self.angle_trajectory_diff = angle_trajectory_diff
        self.normalized_trial_time = normalized_trial_time
        self.unique_rock_id = unique_rock_id

    def __str__(self):
        return f'Livello {self.level_name}, giocatore {self.player_id}, verso pietra piccola: {self.neigh_small}, distanza goal: {self.distance}, angolo goal {self.angle_goal/np.pi*180}, angolo traiettoria {self.angle_trajectory/np.pi*180}'


def create_decision_point_data(player_data: List[PlayerLevelResult], level_data: List[Level]):

    # Per ogni mappa prendo tutti i decision point
    # Per ognuno di essi prendo tutti i giocatori che ci passano
    # Poi calcolo:
    # 1) Tempo quando lo raggiungono
    # 2) Per la traiettoria successiva scelta
    #    - Dimensione pietra successiva
    #    - Angolo tra pietra successiva e goal
    #    - Distanza dal goal in salti
    # 3) Angolo rispetto a piattaforma precedente
    decision_points = []

    for l, level in enumerate(level_data):

        name = level.level_name

        for player in player_data:

            level_results = player.level_results[name]
            trajectory = level_results.trajectory_platforms

            for decision_point in level.decision_points:

                if decision_point.platform.id in trajectory:

                    idx = trajectory.index(decision_point.platform.id)

                    # Controllo che il gioco non si fermi al decision point
                    if idx != len(trajectory) - 1:

                        # Tempo in cui arriva al decision point
                        time = level_results.time_platforms[idx]

                        # Normalizzo anche questo tempo rispetto all'ultimo 
                        normalized_trial_time = time / (level_results.events[-1].time - level_results.events[0].time)

                        next_idx = trajectory[idx + 1]

                        euclidean_distance = np.sqrt((level.platform_last.x - decision_point.platform.x)**2 +
                                                     (level.platform_last.y - decision_point.platform.y)**2)

                        # Devo anche controllare che stiano facendo una scelta
                        # tra quelle possibili per il decision point
                        neigh_inds = [n.id for n in decision_point.neighs]

                        if len(neigh_inds) == 2:

                            if next_idx in neigh_inds:

                                decision_neigh_idx = neigh_inds.index(next_idx)
                                neigh_platform = decision_point.neighs[decision_neigh_idx]
                                neigh_small = neigh_platform.is_small

                                # CAMBIO TUTTO
                                # INVECE DI decision_neigh_idx pari a quello successivo
                                # nella traiettoria, prendo sempre la pietra piccola
                                if decision_point.neighs[0].is_small: decision_neigh_idx = 0
                                else: decision_neigh_idx = 1
                                
                                # decision_neigh_idx = neigh_inds.index(next_idx)
                                neigh_platform = decision_point.neighs[decision_neigh_idx]
                                # neigh_small = neigh_platform.is_small

                                # Valuto la distanza lungo questo percorso
                                distance = len(decision_point.paths[decision_neigh_idx])

                                # Ora valuto l'angolo con cui vedono il goal
                                dir_neigh = np.array([neigh_platform.x - decision_point.platform.x,
                                                    neigh_platform.y - decision_point.platform.y])
                                
                                dir_goal = np.array([level.platform_last.x - decision_point.platform.x,
                                                    level.platform_last.y - decision_point.platform.y])

                                dir_neigh = dir_neigh / np.sqrt(np.sum(np.square(dir_neigh)))
                                dir_goal = dir_goal / np.sqrt(np.sum(np.square(dir_goal)))

                                angle_goal = np.arccos(np.dot(dir_neigh,dir_goal))

                                # E infine valuto l'angolo rispetto alla direzione da cui arriva
                                old_platform = level.platforms[trajectory[idx - 1]]
                                dir_old = np.array([decision_point.platform.x - old_platform.x,
                                                    decision_point.platform.y - old_platform.y])

                                # Attenzione, se la pietra è la prima in assoluto, non c'è una precedente
                                if idx == 0:
                                    dir_old = np.array([0,1])

                                dir_old = dir_old / np.sqrt(np.sum(np.square(dir_old)))
                                dot_prod =  np.dot(dir_neigh,dir_old) 
                                if dot_prod < -1:
                                    dot_prod = -1

                                angle_trajectory = np.arccos(dot_prod)

                                # Path e traiettorie
                                path = decision_point.paths[decision_neigh_idx]
                                trajectory_real = level_results.trajectory

                                # Differenza di angoli e distanze
                                if decision_neigh_idx == 0: k = 1
                                else: k = 0

                                neigh_platform = decision_point.neighs[k]

                                distance_diff = len(decision_point.paths[k]) - distance
                        
                                dir_neigh = np.array([neigh_platform.x - decision_point.platform.x,
                                                      neigh_platform.y - decision_point.platform.y])
                                
                                dir_neigh = dir_neigh / np.sqrt(np.sum(np.square(dir_neigh)))
                                angle_goal_diff = np.arccos(np.dot(dir_neigh, dir_goal)) - angle_goal
                                      
                                if np.dot(dir_neigh, dir_old) < -1 or np.dot(dir_neigh, dir_old) > 1:
                                    continue

                                angle_trajectory_diff = np.arccos(np.dot(dir_neigh, dir_old)) - angle_trajectory
                                # if angle_trajectory_diff > np.pi/2:
                                    # print('Angolo pietra piccola = ', np.arccos(np.dot(dir_neigh, dir_old))/np.pi*180)
                                    # print('Angolo pietra grande = ', angle_trajectory/np.pi*180)

                                # Ultima cosa: identificatore univoco per la pietra
                                unique_rock_id = l * 100 + decision_point.platform.id

                                decision_points.append(DecisionPointData(decision_point, time, neigh_small, 
                                                                        distance, euclidean_distance, angle_goal, angle_trajectory, 
                                                                        name, player.prolific_ID, path,
                                                                        trajectory, trajectory_real, distance_diff,
                                                                        angle_goal_diff, angle_trajectory_diff,
                                                                        normalized_trial_time, unique_rock_id))
      
    return decision_points





def find_closest_platform_to_point(level, point):

    x, y, z = point
    dist_min = 1e10
    ind_min = -1

    for i, platform in enumerate(level.platforms):
        d = np.sqrt((platform.x - x)**2 + (platform.y - z)**2)
        if d < dist_min:
            dist_min = d
            ind_min = i

    return ind_min


def find_platform_under_player(level, point):

    x, y, z = point
    ind_min = -1

    for i, platform in enumerate(level.platforms):
        
        if platform.is_small:
            # radius = 0.575
            radius = 0.95
        else:
            # radius = 0.85
            radius = 1.3
        
        d = np.sqrt((platform.x - x)**2 + (platform.y - z)**2)

        if d <= radius:
            return i

    return ind_min


def get_trajectory_times(player_data, level, player_id):

    times = player_data.time_interp
    trajectory = player_data.trajectory_interp
    in_water = player_data.is_in_water
    angles = player_data.angles_interp
    
    T = times.shape[0]

    jump_lim = 0.3
    first_platform_ymin = 0.7
    first_platform_ymax = 0.85
    rotation_threshold = 0.001
    movement_threshold = 0.005
    velocity_y_threshod = 0.01

    velocity_x = np.diff(trajectory[:,0], prepend=trajectory[0,0])
    velocity_z = np.diff(trajectory[:,2], prepend=trajectory[0,2])
    velocity = np.sqrt(velocity_x**2 + velocity_z**2)
    velocity_y = np.diff(trajectory[:,1], prepend=trajectory[0,1])

    angular_speed = np.diff(angles, prepend=angles[0])
    first_platform_z = level.platforms[0].y

    # Array in cui colleziono le informazioni
    data = pd.DataFrame(
        columns=['Level','Player','Time','PlatformInd','PlatformType','Moving','Rotating']
    )

    # PlatformType = FirstDecision, Decision, Predecision, NonDecision

    # Ciclo su tutti gli istanti
    for i in range(T):
        
        # Prendo le coordinate dell'istante attuale e precedente
        _, y, z = trajectory[i, :]

        # Identifico la piattaforma, se c'è, sotto il player
        platform_ind = find_platform_under_player(level, trajectory[i, :])
        platform_type = 'Undefined'

        if platform_ind == -1:

            # Capisco se sono sulla primissima piattaforma
            # Cioè se sono prima dalla piattaforma 0 lungo Z e se sono più in alto di un tot, ma sotto un altro tot

            if z < first_platform_z - 1:
                if y >= first_platform_ymin and y <= first_platform_ymax:
                    platform_type = 'FirstPlatform'
        else:
            
            if platform_ind in level.decision_ids:
                platform_type = 'Decision'
            elif platform_ind in level.predecision_ids:
                platform_type = 'Predecision'
            else:
                platform_type = 'Nondecision'
        
        # Inizializzo
        grounded = False
        moving = False
        rotating = False
            
        if y <= jump_lim and np.abs(velocity_y[i]) <= velocity_y_threshod:
            grounded = True

        if platform_type == 'FirstPlatform': grounded = True

        if velocity[i] > movement_threshold:
            moving = True

        if np.abs(angular_speed[i]) > rotation_threshold:
            rotating = True

        if platform_type != 'Undefined' and grounded is True and in_water[i] != 1:

            # Aggiungo questo punto
            data.loc[len(data)] = pd.Series(
                {'Level': level.level_name, 'Player': player_id, 'Time': times[i], 'PlatformInd': platform_ind,
                 'PlatformType': platform_type, 'Moving': moving, 'Rotating': rotating})

    return data





def get_trajectory_times_alternative(player_data, level, player_id):

    times = player_data.time_interp
    trajectory = player_data.trajectory_interp
    in_water = player_data.is_in_water
    angles = player_data.angles_interp
    velocity = player_data.velocity_interp
    angular_velocity = player_data.angular_velocity_interp
    velocity_xz = np.sqrt(velocity[:,0]**2 + velocity[:,2]**2)
    in_water = player_data.is_in_water

    T = times.shape[0]

    velocity_y_threshold = 0.01
    height_threshold = 0.6
    rotation_threshold = 0.02
    movement_threshold = 0.025
    
    traj_platforms = np.unique(player_data.trajectory_platforms)

    grounded = True
    rotating = False
    moving = False
    was_in_water = False

    on_ground = np.zeros(T)
    is_rotating = np.zeros(T)
    is_moving = np.zeros(T)
    water = np.zeros(T)
    platforms_ind = np.zeros(T)

    last_platform_ind = -1
    platform_count = 0
    

    # Ciclo su tutti i tempi e cerco di capire in ch stato sono
    for t in range(T):

        p_ind = find_closest_platform_to_point(level, trajectory[t,:])

        # Ogni volta che vy supera un certo threshold positivo suppongo
        if grounded:
            if velocity[t,1] > velocity_y_threshold:
                grounded = False
                was_in_water = False

        else:
            if np.abs(velocity[t,1]) < velocity_y_threshold and trajectory[t,1] < height_threshold:
                grounded = True
                if last_platform_ind != p_ind:
                    last_platform_ind = p_ind
                    platform_count += 1
            elif velocity[t-1,1] < -velocity_y_threshold and velocity[t,1] > velocity_y_threshold:
                grounded = True

                if last_platform_ind != p_ind:
                    last_platform_ind = p_ind
                    platform_count += 1


        if np.abs(angular_velocity[t]) > rotation_threshold:
            rotating = True
        else:
            rotating = False

        if velocity_xz[t] > movement_threshold:
            moving = True
        else:
            moving = False

        if in_water[t-1] > 0.5 and in_water[t] < 0.5:
            was_in_water = True

        is_or_was_in_water = was_in_water | bool(in_water[t])
        
        on_ground[t] = grounded
        is_rotating[t] = rotating
        is_moving[t] = moving
        water[t] = is_or_was_in_water
        platforms_ind[t] = last_platform_ind

    platforms_ind_unique = np.unique(platforms_ind)

    if len(platforms_ind_unique) != len(traj_platforms) + 1:
        print(f'Livello {level.level_name}, soggetto: {player_id}, errore nel conteggio delle piattaforme (previste = {len(traj_platforms)}, contate = {len(platforms_ind_unique) - 1})')

    # Ora converto queste serie in tempi per un dizionario
    # Prima però creo le serie combinate che mi interessano
    on_ground_still = on_ground * (1 - is_moving) * (1-is_rotating) * (1-water)
    on_ground_only_rotation = on_ground * (1 - is_moving)  * (1-water)
    on_ground_moving = on_ground * (1-water)

    time_results = dict()

    for p_ind in platforms_ind_unique:
           
        mask = np.zeros(T)
        mask[p_ind == platforms_ind] = 1

        platf_ground_still = on_ground_still * mask
        platf_ground_only_rotation = on_ground_only_rotation * mask
        platf_ground_moving = on_ground_moving * mask

        time_results[int(p_ind)] = {'Still': np.sum(platf_ground_still)/60, 'OnlyRotation': np.sum(platf_ground_only_rotation)/60, 'Moving': np.sum(platf_ground_moving)/60}

    return time_results