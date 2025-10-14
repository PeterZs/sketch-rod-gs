import time

import dijkstra_cpp
import making_correspond
import matplotlib.pyplot as plt
import numpy as np
import preprocess_cpp


def connect_small_polyline(
    points: np.ndarray,
    most_contribute_id: np.ndarray,
    prev_map: list[list[tuple[int, int]]],
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
) -> list[np.ndarray]:
    polyline_list = []

    id_set = set()  # Avoid including same point

    prev_x = end_x
    prev_y = end_y

    while prev_x != -1 or prev_y != -1:
        id = most_contribute_id[prev_y][prev_x]
        if id not in id_set:
            # self.pc._features_dc[id, 0, :] = torch.tensor([2.0, 0.0, 2.0])
            polyline_list.append(points[id])
            id_set.add(id)
        if prev_x == start_x and prev_y == start_y:
            break
        prev_x, prev_y = prev_map[prev_y][prev_x]

    return polyline_list


def make_small_polyline(
    drawed_path_segment: list[tuple[int, int]],
    points: np.ndarray,
    most_contribute_id: np.ndarray,
    image: np.ndarray,
    strip_ids: np.ndarray,
    strip_dists: np.ndarray,
    strip_times: np.ndarray,
    W: int,
    H: int,
    start_segment_id: int,
    dist_3d_threshhold: float,
    debug: bool,
) -> tuple[list[np.ndarray], list, int, float, tuple[int, int]]:
    if debug:
        print("make_small_polyline is called")

    start_point = drawed_path_segment[start_segment_id]
    start_x = int(start_point[0])
    start_y = int(start_point[1])

    goal_point = drawed_path_segment[-1]
    goal_x = int(goal_point[0])
    goal_y = int(goal_point[1])

    prev_map, head_segment_id, head_segment_time, head_segment_pixel, reach_goal = dijkstra_cpp.dijkstra(
        points.astype(np.float32),
        most_contribute_id.astype(np.int32),
        image.astype(np.uint8),
        strip_ids,
        strip_dists,
        strip_times,
        (start_x, start_y),
        (goal_x, goal_y),
        dist_3d_threshhold,
        W,
        H,
        debug,
    )

    # Goal
    if reach_goal:
        segment_end_x = goal_x
        segment_end_y = goal_y
    # Doesn't goal
    else:
        segment_end_x = head_segment_pixel[0]
        segment_end_y = head_segment_pixel[1]

    polyline_segment = connect_small_polyline(
        points=points,
        most_contribute_id=most_contribute_id,
        prev_map=prev_map,
        start_x=start_x,
        start_y=start_y,
        end_x=segment_end_x,
        end_y=segment_end_y,
    )

    return polyline_segment, prev_map, head_segment_id, head_segment_time, head_segment_pixel, reach_goal


def connect_polyline_segment(small_polyline_list: list[list[np.ndarray]]):
    selected_small_polyline_list = [small_polyline_list[0]]
    i = 0
    while i < len(small_polyline_list) - 1:
        before_end_point = small_polyline_list[i][-1]
        min_dist = np.linalg.norm(small_polyline_list[i + 1][0] - before_end_point)
        min_dist_idx = i + 1
        for j in range(i + 1, len(small_polyline_list)):
            next_first_point = small_polyline_list[j][0]
            dist = np.linalg.norm(next_first_point - before_end_point)
            if dist < min_dist:
                min_dist = dist
                min_dist_idx = j
        selected_small_polyline_list.append(small_polyline_list[min_dist_idx])
        i = min_dist_idx + 1

    ret = []
    for selected_small_polyline in selected_small_polyline_list:
        ret += selected_small_polyline

    return ret


def make_polyline_segment(
    drawed_path_segment: list[tuple[int, int]],
    points: np.ndarray,
    most_contribute_id: np.ndarray,
    image: np.ndarray,
    strip_ids: np.ndarray,
    strip_dists: np.ndarray,
    strip_times: np.ndarray,
    dist_3d_threshhold: float,
    W: int,
    H: int,
    debug: bool,
):
    small_polyline_list: list[list[np.ndarray]] = []

    head_segment_id = 0

    if debug:
        print("len(drawed_path_segment): ", len(drawed_path_segment))

    reach_segment_goal = False
    while reach_segment_goal is False:
        if debug:
            print("head_segment_id: ", head_segment_id)

        small_polyline, prev_map, next_head_segment_id, head_segment_time, head_segment_pixel, reach_goal = (
            make_small_polyline(
                drawed_path_segment=drawed_path_segment,
                points=points,
                most_contribute_id=most_contribute_id,
                image=image,
                strip_ids=strip_ids,
                strip_dists=strip_dists,
                strip_times=strip_times,
                W=W,
                H=H,
                start_segment_id=head_segment_id,
                dist_3d_threshhold=dist_3d_threshhold,
                debug=debug,
            )
        )

        small_polyline.reverse()  # 後ろから探索しているので反対にする

        if debug:
            print("\tsmall_polyline: ", small_polyline)
            print("\tnext_head_segment_id: ", next_head_segment_id)
            print("\thead_segment_time: ", head_segment_time)
            print("\thead_segment_pixel: ", head_segment_pixel)
            print("\treach_goal: ", reach_goal)

        small_polyline_list.append(small_polyline)

        head_segment_id = next_head_segment_id + 1

        reach_segment_goal = reach_goal

    return connect_polyline_segment(small_polyline_list)
    # return small_polyline_list


def visualize_strip_id_and_distance(
    strip_ids, strip_dists, line_time_list, out_depth_of_most_contribute, W: int, H: int
):
    normed_depth = out_depth_of_most_contribute / np.max(out_depth_of_most_contribute)
    plt.imsave("depth_map.png", normed_depth, cmap="gray")
    plt.imsave("strip_dists.png", strip_dists, cmap="gray")
    id_out = np.zeros((H, W, 3), dtype=np.float32)
    cmap = plt.get_cmap("plasma")
    for y in range(H):
        for x in range(W):
            id = strip_ids[y][x]
            if id == -1.0:
                continue
            t = line_time_list[id]
            color = cmap(t)[:3]  # RGBA のうち RGB を返す
            id_out[y, x] = color
    plt.imsave("strip_ids.png", id_out)


def cerate_line(
    W: int,
    H: int,
    drawed_path: list[tuple[int, int]],
    points: np.ndarray,
    most_contribute_id: np.ndarray,
    out_depth_of_most_contribute: np.ndarray,
    image: np.ndarray,
    tube_radius: float,
    debug: bool,
) -> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    if debug:
        t = time.perf_counter()
    strip_ids, strip_dists, strip_times, line_time_list, line_len, modified_drawed_path = preprocess_cpp.preprocess(
        W, H, drawed_path, out_depth_of_most_contribute, tube_radius, debug
    )
    if debug:
        print("cpp preprocess time: ", time.perf_counter() - t)

    # visualize_strip_id_and_distance(
    #     strip_ids=strip_ids, strip_dists=strip_dists, line_time_list=line_time_list,
    #     out_depth_of_most_contribute=out_depth_of_most_contribute, W=W, H=H
    # )

    if debug:
        print("done preprocess")

    def get_gradient_color(t):
        """
        0〜1の範囲の値 t に対して、きれいなグラデーションカラー(R, G, B)を返す。
        """
        # cmap = plt.get_cmap('viridis')  # 他に 'plasma', 'inferno', 'magma' なども美しい
        cmap = plt.get_cmap("plasma")
        color = cmap(t)
        return color[:3]  # RGBA のうち RGB を返す

    color_list = [get_gradient_color(time) for time in line_time_list]

    if debug:
        print("len(drawed_path): ", len(drawed_path))

    t = time.perf_counter()
    # polyline_list = []
    # for i, drawed_path_segment in enumerate(splitted_drawed_path):
    #     if i != 0 and i != len(splitted_drawed_path) - 1:
    #         discard_sample_num = 5
    #         drawed_path_segment = drawed_path_segment[discard_sample_num : -(discard_sample_num + 1)]

    #     polyline_segment = make_polyline_segment(
    #         drawed_path_segment=drawed_path_segment,
    #         points=points,
    #         most_contribute_id=most_contribute_id,
    #         image=image,
    #         strip_ids=strip_ids,
    #         strip_dists=strip_dists,
    #         strip_times=strip_times,
    #         dist_3d_threshhold=tube_radius * 3,
    #         W=W,
    #         H=H,
    #     )

    #     polyline_list += polyline_segment

    #     print("Tracing path done")

    polyline_segment = make_polyline_segment(
        drawed_path_segment=modified_drawed_path,
        points=points,
        most_contribute_id=most_contribute_id,
        image=image,
        strip_ids=strip_ids,
        strip_dists=strip_dists,
        strip_times=strip_times,
        dist_3d_threshhold=tube_radius * 3,
        W=W,
        H=H,
        debug=debug,
    )

    if debug:
        print("all dijkstra time: ", time.perf_counter() - t)

    polyline = np.array(polyline_segment)

    # Resampling
    if debug:
        t = time.perf_counter()
    sampling_rate = 0.05  # Sampling rate
    stroke = [polyline[0]]
    jcur = 0
    rcur = 0
    lcur = sampling_rate
    while True:
        if jcur >= polyline.shape[0] - 1:
            break
        lenj = np.linalg.norm(polyline[jcur + 1] - polyline[jcur])
        lenjr = lenj * (1.0 - rcur)
        if lenjr > lcur:
            rcur += lcur / lenj
            stroke.append((1 - rcur) * polyline[jcur] + rcur * polyline[jcur + 1])
            lcur = sampling_rate
        else:
            lcur -= lenjr
            rcur = 0
            jcur += 1
    polyline = np.array(stroke)
    if debug:
        print("resampling time: ", time.perf_counter() - t)

    # Smoothing
    if debug:
        t = time.perf_counter()
    for _ in range(2):
        alpha = 0.5
        prev_polyline = polyline.copy()
        for i, point in enumerate(prev_polyline):
            if i == 0 or i == polyline.shape[0] - 1:
                continue
            before = prev_polyline[i - 1]
            after = prev_polyline[i + 1]
            delta = before + after - 2 * point
            polyline[i] += alpha * delta
    if debug:
        print("smoothing time: ", time.perf_counter() - t)

    # Construct correspondance between primitives and polyline
    if debug:
        t = time.perf_counter()
    primitive_binding_id, primitive_binding_time = making_correspond.making_correspondance(
        points, polyline, tube_radius
    )
    if debug:
        print("Constructing correspondance time: ", time.perf_counter() - t)

    return color_list, primitive_binding_id, primitive_binding_time, polyline
