import math


def rotate_rect(x1, y1, x2, y2, degree, center_x, center_y):
    points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    new_points = list()
    for point in points:
        dx = point[0] - center_x
        dy = point[1] - center_y
        new_x = center_x + dx * math.cos(degree) - dy * math.sin(degree)
        new_y = center_y + dx * math.sin(degree) + dy * math.cos(degree)
        new_points.append([(new_x), (new_y)])
    return new_points
