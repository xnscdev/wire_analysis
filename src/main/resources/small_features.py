import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from shapely.geometry import Point, Polygon
import sys


def poly_coords(p):
    return np.array(list(zip(*p.exterior.coords.xy)))


def bound_contours(bound):
    return np.array([[bound[i] for i in range(len(bound))]], dtype=np.int32),


def poly_contours(p):
    return bound_contours(poly_coords(p))


def unit_vec(v):
    return v / np.linalg.norm(v)


def angle(v1, v2):
    u1 = unit_vec(v1)
    u2 = unit_vec(v2)
    return np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0))


def bounds(p):
    rect = poly_coords(p.minimum_rotated_rectangle)
    d1 = Point(rect[0, 0], rect[0, 1]).distance(Point(rect[1, 0], rect[1, 1]))
    d2 = Point(rect[1, 0], rect[1, 1]).distance(Point(rect[2, 0], rect[2, 1]))
    if d1 < d2:
        return d2, d1
    else:
        return d1, d2


def flood(i, sx, sy, v, c, pm):
    a = 0
    fill = set()
    fill.add((sx, sy))
    while fill:
        (px, py) = fill.pop()
        if px < 0 or px >= i.shape[0] or py < 0 or py >= i.shape[1]:
            continue
        if v[px, py] or i[px, py]:
            continue
        v[px, py] = True
        c[px, py] = 255
        pm.add((px, py))
        a += 1
        fill.add((px + 1, py))
        fill.add((px - 1, py))
        fill.add((px, py + 1))
        fill.add((px, py - 1))
    return a


def main(args):
    img = cv2.imread(args[1], 0)
    small_dmap = np.zeros(img.shape, dtype=np.float64)
    particles = []
    visited = np.zeros(img.shape, dtype=bool)
    ci = np.zeros(img.shape, dtype=np.uint8)
    rem = np.zeros(img.shape, dtype=np.uint8)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            pixels = set()
            area = flood(img, x, y, visited, ci, pixels)
            if area:
                if area < 1000:
                    coa, _ = cv2.findContours(ci, cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_SIMPLE)
                    if len(coa[0]) < 3:
                        continue
                    poly = Polygon([p[0] for p in coa[0]])
                    m1, m2 = bounds(poly)
                    am = np.pi * m1 * m2 / 4
                    diam = np.sqrt(am / np.pi) * 2
                    particles.append(diam)
                    for p in pixels:
                        small_dmap[p[0], p[1]] = diam
                else:
                    rem = cv2.add(rem, ci)
                ci = np.zeros(img.shape, dtype=np.uint8)
    nm_per_pixel = 1000 / int(args[2])
    plt.hist(np.array(particles) * nm_per_pixel)
    plt.title('Small feature size distribution')
    plt.xlabel('Average Diameter (nm)')
    plt.ylabel('Frequency')
    base = os.path.join(args[3], os.path.basename(args[1]).replace('_median.tif', ''))
    plt.savefig(base + '_small_dist.tif', bbox_inches='tight')
    with open(base + '_small_features.npy', 'wb') as f:
        np.save(f, small_dmap)
    cv2.imwrite(base + '_wires.tif', rem)


if __name__ == '__main__':
    main(sys.argv)
