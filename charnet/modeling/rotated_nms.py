import numpy as np
import pyclipper
from shapely.geometry import Polygon

from tqdm import tqdm as tqdm
import pickle
from time import time

from ctypes import c_float, c_int, c_double, POINTER, byref


def nms(boxes, overlapThresh, neighbourThresh=0.5, minScore=0, num_neig=0):
    start_time = time()
    new_boxes = np.zeros_like(boxes)
    pick = []
    suppressed = [False for _ in range(boxes.shape[0])]
    areas = [Polygon([(b[0], b[1]), (b[2], b[3]), (b[4], b[5]), (b[6], b[7])]).area
             for b in boxes]
    polygons = pyclipper.scale_to_clipper(boxes[:, :8].reshape((-1, 4, 2)))
    centers = [
        [
            (p[0][0] + p[1][0] + p[2][0] + p[3][0]) / 4,
            (p[0][1] + p[1][1] + p[2][1] + p[3][1]) / 4
        ] for p in polygons]
    sides = []
    for p in polygons:
        x_coordinates = [p[0][0], p[1][0], p[2][0], p[3][0]]
        y_coordinates = [p[0][1], p[1][1], p[2][1], p[3][1]]

        side_x = max(x_coordinates) - min(x_coordinates)
        side_y = max(y_coordinates) - min(y_coordinates)
        sides.append([side_x, side_y])

    order = boxes[:, 8].argsort()[::-1]
    print('nms..')
    for _i, i in enumerate(tqdm(order)):
        if suppressed[i] is False:
            pick.append(i)
            neighbours = list()
            for j in order[_i+1:]:
                var_x = ((sides[i][0]+sides[j][0])/2-abs(centers[i][0] - centers[j][0])) > 0
                var_y = ((sides[i][1]+sides[j][1])/2-abs(centers[i][1] - centers[j][1])) > 0
                if var_x and var_y:
                    if suppressed[j] is False:
                        try:
                            pc = pyclipper.Pyclipper()
                            pc.AddPath(polygons[i], pyclipper.PT_CLIP, True)
                            pc.AddPaths([polygons[j]], pyclipper.PT_SUBJECT, True)
                            solution = pc.Execute(pyclipper.CT_INTERSECTION)
                            if len(solution) == 0:
                                inter = 0
                            else:
                                inter = pyclipper.scale_from_clipper(
                                    pyclipper.scale_from_clipper(
                                        pyclipper.Area(solution[0])))
                        except:
                            inter = 0
                        union = areas[i] + areas[j] - inter
                        iou = inter / union if union > 0 else 0
                        if union > 0 and iou > overlapThresh:
                            suppressed[j] = True
                        if iou > neighbourThresh:
                            neighbours.append(j)
            if len(neighbours) >= num_neig:
                neighbours.append(i)
                temp_scores = (boxes[neighbours, 8] - minScore).reshape((-1, 1))
                new_boxes[i, :8] = (boxes[neighbours, :8] * temp_scores).sum(axis=0) / temp_scores.sum()
                new_boxes[i, 8] = boxes[i, 8]
            else:
                for ni in neighbours:
                    suppressed[ni] = False
                pick.pop()

    assert_dict = {
        'input': [boxes, overlapThresh, neighbourThresh, minScore, num_neig],
        'output': [pick, new_boxes],
        'time': time()-start_time
    }

    return pick, new_boxes


def cpp_nms(
        cpp_nms_ptr,
        boxes: np.ndarray,
        overlapThresh: float,
        neighbourThresh=0.5, minScore=0, num_neig=0
):
    n, m = boxes.shape
    new_boxes = np.zeros_like(boxes).astype(np.float32)
    pick = (np.zeros(n) - 1).astype(np.int)

    c_float_p = POINTER(c_float)
    c_int_p = POINTER(c_int)

    boxes_p = boxes.astype(np.float32).ctypes.data_as(c_float_p)
    n_c = c_int(n)
    m_c = c_int(m)

    new_boxes_p = new_boxes.ctypes.data_as(c_float_p)
    pick_p = pick.ctypes.data_as(c_int_p)

    overlapThresh_c = c_double(overlapThresh)
    neighbourThresh_c = c_double(neighbourThresh)
    minScore_c = c_float(minScore)
    num_neig_c = c_int(num_neig)

    res = cpp_nms_ptr(
        boxes_p, byref(n_c), byref(m_c),
        new_boxes_p, pick_p,
        byref(overlapThresh_c), byref(neighbourThresh_c), byref(minScore_c), byref(num_neig_c)
    )

    if res != 0:
        raise RuntimeError('Dude, sumting wong with C++ func')

    new_boxes = np.array(new_boxes_p[:n * m]).reshape(n, m)
    pick = pick_p[:n]
    pick = [p for p in pick if p != -1]

    return pick, new_boxes


def nms_with_char_cls(boxes, char_scores, overlapThresh, neighbourThresh=0.5, minScore=0, num_neig=0):
    new_boxes = np.zeros_like(boxes)
    new_char_scores = np.zeros_like(char_scores)
    pick = []
    suppressed = [False for _ in range(boxes.shape[0])]
    areas = [Polygon([(b[0], b[1]), (b[2], b[3]), (b[4], b[5]), (b[6], b[7])]).area
             for b in boxes]
    polygons = pyclipper.scale_to_clipper(boxes[:, :8].reshape((-1, 4, 2)))

    centers = [
        [
            (p[0][0] + p[1][0] + p[2][0] + p[3][0]) / 4,
            (p[0][1] + p[1][1] + p[2][1] + p[3][1]) / 4
        ] for p in polygons]
    sides = []
    for p in polygons:
        x_coordinates = [p[0][0], p[1][0], p[2][0], p[3][0]]
        y_coordinates = [p[0][1], p[1][1], p[2][1], p[3][1]]

        side_x = max(x_coordinates) - min(x_coordinates)
        side_y = max(y_coordinates) - min(y_coordinates)
        sides.append([side_x, side_y])

    order = boxes[:, 8].argsort()[::-1]
    print('nms with char cls...')
    for _i, i in enumerate(tqdm(order)):
        if suppressed[i] is False:
            pick.append(i)
            neighbours = list()
            for j in order[_i+1:]:
                var_x = ((sides[i][0]+sides[j][0])/2-abs(centers[i][0] - centers[j][0])) > 0
                var_y = ((sides[i][1]+sides[j][1])/2-abs(centers[i][1] - centers[j][1])) > 0
                if var_x and var_y and (suppressed[j] is False):
                    pc = pyclipper.Pyclipper()
                    pc.AddPath(polygons[i], pyclipper.PT_CLIP, True)
                    pc.AddPaths([polygons[j]], pyclipper.PT_SUBJECT, True)
                    solution = pc.Execute(pyclipper.CT_INTERSECTION)
                    if len(solution) == 0:
                        inter = 0
                    else:
                        inter = pyclipper.scale_from_clipper(
                            pyclipper.scale_from_clipper(
                                pyclipper.Area(solution[0])))
                    union = areas[i] + areas[j] - inter
                    iou = inter / union if union > 0 else 0
                    if union > 0 and iou > overlapThresh:
                        suppressed[j] = True
                    if iou > neighbourThresh:
                        neighbours.append(j)
            if len(neighbours) >= num_neig:
                neighbours.append(i)
                temp_scores = (boxes[neighbours, 8] - minScore).reshape((-1, 1))
                new_boxes[i, :8] = (boxes[neighbours, :8] * temp_scores).sum(axis=0) / temp_scores.sum()
                new_boxes[i, 8] = boxes[i, 8]
                new_char_scores[i, :] = (char_scores[neighbours, :] * temp_scores).sum(axis=0) / temp_scores.sum()
            else:
                for ni in neighbours:
                    suppressed[ni] = False
                pick.pop()

    return pick, new_boxes, new_char_scores


def cpp_nms_with_char_cls(
        cpp_nms_with_char_cls_ptr,
        char_boxes: np.ndarray,
        char_scores: np.ndarray,
        overlapThresh: float,
        neighbourThresh=0.5, minScore=0, num_neig=0
):
    n, m = char_boxes.shape
    new_char_boxes = np.zeros_like(char_boxes).astype(np.float32)
    pick = (np.zeros(n) - 1).astype(np.int)
    nc, mc = char_scores.shape
    new_char_scores = np.zeros_like(char_scores).astype(np.float32)

    c_float_p = POINTER(c_float)
    c_int_p = POINTER(c_int)

    char_boxes_p = char_boxes.astype(np.float32).ctypes.data_as(c_float_p)
    n_c = c_int(n)
    m_c = c_int(m)

    new_char_boxes_p = new_char_boxes.ctypes.data_as(c_float_p)
    pick_p = pick.ctypes.data_as(c_int_p)

    char_scores_p = char_scores.astype(np.float32).ctypes.data_as(c_float_p)
    nc_c = c_int(nc)
    mc_c = c_int(mc)
    new_char_scores_p = new_char_scores.ctypes.data_as(c_float_p)

    overlapThresh_c = c_double(overlapThresh)
    neighbourThresh_c = c_double(neighbourThresh)
    minScore_c = c_float(minScore)
    num_neig_c = c_int(num_neig)

    res = cpp_nms_with_char_cls_ptr(
        char_boxes_p, byref(n_c), byref(m_c),
        new_char_boxes_p, pick_p,
        char_scores_p, byref(nc_c), byref(mc_c),
        new_char_scores_p,
        byref(overlapThresh_c), byref(neighbourThresh_c), byref(minScore_c), byref(num_neig_c)
    )

    if res != 0:
        raise RuntimeError('Dude, sumting wong with C++ func')

    new_char_boxes = np.array(new_char_boxes_p[:n * m]).reshape(n, m)
    new_char_scores = np.array(new_char_scores_p[:nc * mc]).reshape(nc, mc)
    pick = pick_p[:n]
    pick = [p for p in pick if p != -1]

    return pick, new_char_boxes, new_char_scores


def softnms(boxes, box_scores, char_scores=None, overlapThresh=0.3,
                          threshold=0.8, neighbourThresh=0.5, num_neig=0):
    scores = box_scores.copy()
    new_boxes = boxes[:, 0: 8].copy()
    if char_scores is not None:
        new_char_scores = char_scores.copy()
    polygons = [pyclipper.scale_to_clipper(poly.reshape((-1, 2))) for poly in new_boxes]
    areas = [pyclipper.scale_from_clipper(pyclipper.scale_from_clipper(
             pyclipper.Area(poly))) for poly in polygons]
    areas = [abs(_) for _ in areas]
    N = boxes.shape[0]
    order = np.arange(N)
    i = 0
    while i < N:
        max_pos = scores[order[i: N]].argmax() + i
        order[i], order[max_pos] = order[max_pos], order[i]
        pos = i + 1
        neighbours = list()
        while pos < N:
            try:
                pc = pyclipper.Pyclipper()
                pc.AddPath(polygons[order[i]], pyclipper.PT_CLIP, True)
                pc.AddPaths([polygons[order[pos]]], pyclipper.PT_SUBJECT, True)
                solution = pc.Execute(pyclipper.CT_INTERSECTION)
                if len(solution) == 0:
                    inter = 0
                else:
                    inter = pyclipper.scale_from_clipper(
                        pyclipper.scale_from_clipper(
                            pyclipper.Area(solution[0])))
            except Exception:
                inter = 0
            union = areas[order[i]] + areas[order[pos]] - inter
            iou = inter / union if union > 0 else 0
            if iou > neighbourThresh:
                neighbours.append(order[pos])
            weight = np.exp(-(iou **2) / 0.5)
            scores[order[pos]] *= weight
            if scores[order[pos]] < threshold:
                order[pos], order[N - 1] = order[N - 1], order[pos]
                N -= 1
                pos -= 1
            pos += 1
        if len(neighbours) >= num_neig:
            neighbours.append(order[i])
            temp_scores = box_scores[neighbours].reshape((-1, 1))
            new_boxes[order[i], :8] = (boxes[neighbours, :8] * temp_scores).sum(axis=0) / temp_scores.sum()
            if char_scores is not None:
                new_char_scores[order[i], :] = (char_scores[neighbours, :] * temp_scores).sum(axis=0) / temp_scores.sum()
        else:
            order[i], order[N - 1] = order[N - 1], order[i]
            N -= 1
            i -= 1
        i += 1
    keep = [order[_] for _ in range(N)]
    if char_scores is not None:
        return keep, new_boxes, new_char_scores
    else:
        return keep, new_boxes


def nms_poly(polys, scores, overlapThresh, neighbourThresh=0.5, minScore=0, num_neig=0):
    pick = list()
    suppressed = [False for _ in range(len(polys))]
    polygons = [pyclipper.scale_to_clipper(poly.reshape((-1, 2))) for poly in polys]
    areas = [pyclipper.scale_from_clipper(pyclipper.scale_from_clipper(
             pyclipper.Area(poly))) for poly in polygons]
    areas = [abs(_) for _ in areas]
    order = np.array(scores).argsort()[::-1]
    for _i, i in enumerate(order):
        if suppressed[i] is False:
            pick.append(i)
            neighbours = list()
            for j in order[_i+1:]:
                if suppressed[j] is False:
                    try:
                        pc = pyclipper.Pyclipper()
                        pc.AddPath(polygons[i], pyclipper.PT_CLIP, True)
                        pc.AddPaths([polygons[j]], pyclipper.PT_SUBJECT, True)
                        solution = pc.Execute(pyclipper.CT_INTERSECTION)
                        if len(solution) == 0:
                            inter = 0
                        else:
                            inter = pyclipper.scale_from_clipper(
                                pyclipper.scale_from_clipper(
                                    pyclipper.Area(solution[0])))
                    except Exception as e:
                        inter = 0
                    union = areas[i] + areas[j] - inter
                    iou = inter / union if union > 0 else 0
                    if union > 0 and iou > overlapThresh:
                        suppressed[j] = True
                    if iou > neighbourThresh:
                        neighbours.append(j)
            if len(neighbours) < num_neig:
                for ni in neighbours:
                    suppressed[ni] = False
                pick.pop()
    return pick