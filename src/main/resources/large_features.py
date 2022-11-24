import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import shapely.affinity as affinity
from shapely.geometry import LineString, Point, Polygon
import shutil
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
    img = cv2.imread(args[1] + 'wires_seg.tif', 0)
    iters = int(args[2])
    extra_iters = int(args[3])
    particles = []
    if iters < 0:
        eroded = cv2.dilate(img, (5, 5), iterations=-iters)
    else:
        eroded = cv2.erode(img, (5, 5), iterations=iters)
    dmap = np.zeros(img.shape, dtype=np.float64)
    inverted = cv2.bitwise_not(eroded)
    visited = np.zeros(img.shape, dtype=bool)
    ci = np.zeros(img.shape, dtype=np.uint8)
    r = 0
    img_dir = args[1] + 'wires'
    try:
        shutil.rmtree(img_dir)
    except FileNotFoundError:
        pass
    os.mkdir(img_dir)
    for x in range(inverted.shape[0]):
        for y in range(inverted.shape[1]):
            pixels = set()
            area = flood(inverted, x, y, visited, ci, pixels)
            if area:
                if area > 1000:
                    r += 1
                    if iters < 0:
                        ci = cv2.erode(ci, (5, 5), iterations=-iters)
                    else:
                        ci = cv2.dilate(ci, (5, 5), iterations=iters)
                    ci = cv2.dilate(ci, (5, 5), iterations=15)
                    ci = cv2.erode(ci, (5, 5), iterations=15)
                    ci = cv2.dilate(ci, (5, 5), iterations=extra_iters)
                    c, _ = cv2.findContours(ci, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    cci = cv2.cvtColor(ci, cv2.COLOR_GRAY2BGR)
                    cinr = cci[:]
                    cv2.imwrite(f'{img_dir}/{r}.tif', cci)
                    best = None
                    bestsize = 0
                    for ic in c:
                        if len(ic) < 3:
                            continue
                        poly = Polygon([p[0] for p in ic])
                        poly = poly.simplify(3)
                        d1, d2 = bounds(poly)
                        if d1 * d2 > bestsize:
                            best = poly
                            bestsize = d1 * d2

                    bound = poly_coords(best.minimum_rotated_rectangle).astype(np.int32)
                    cv2.drawContours(cinr, bound_contours(bound), -1, (255, 0, 0), 5)
                    cv2.drawContours(cinr, poly_contours(best), -1, (0, 0, 255), 5)
                    cv2.imwrite(f'{img_dir}/{r}_boxed.tif', cinr)
                    a = angle(bound[1] - bound[0], (1, 0))
                    poly = affinity.rotate(best, a, use_radians=True)
                    bound = poly_coords(poly.minimum_rotated_rectangle).astype(np.int32)
                    # Some of them rotate the wrong way, correct those. Run
                    # this a few times to help correct floating-point error.
                    for i in range(3):
                        if np.abs(bound[1, 1] - bound[0, 1]) > 1:
                            poly = affinity.rotate(best, -a, use_radians=True)
                            bound = poly_coords(poly.minimum_rotated_rectangle).astype(np.int32)
                            a = angle(bound[1] - bound[0], (1, 0))

                    # Make longer side aligned to X axis
                    xa = bound[1, 0] - bound[0, 0]
                    ya = bound[2, 1] - bound[1, 1]
                    if ya > xa:
                        poly = affinity.rotate(poly, 90)
                        bound = poly_coords(poly.minimum_rotated_rectangle).astype(np.int32)

                    # Some polygons are in the wrong order, fix them
                    if bound[0, 0] > bound[2, 0]:
                        bound = np.array([bound[3], bound[0], bound[1], bound[2], bound[3]])

                    # Move polygon to origin
                    poly = affinity.translate(poly, -bound[0, 0], -bound[0, 1])
                    bound = poly_coords(poly.minimum_rotated_rectangle).astype(np.int32)
                    cv2.drawContours(cci, bound_contours(bound), -1, (255, 0, 0), 5)
                    cv2.drawContours(cci, poly_contours(poly), -1, (0, 0, 255), 5)
                    cv2.imwrite(f'{img_dir}/{r}_trans_rot.tif', cci)

                    end_x = bound[1, 0]
                    end_y = bound[2, 1]
                    diams = []
                    for bx in range(end_x):
                        # Calculate intersection length to use for average
                        cut = LineString(((bx, 0), (bx, end_y)))
                        inter = poly.intersection(cut)
                        diam = inter.length
                        if diam:
                            diams.append(diam)

                    diam = np.mean(diams)
                    for p in pixels:
                        dmap[p[0], p[1]] = diam
                    particles.append(diam)
                ci = np.zeros(img.shape, dtype=np.uint8)

    nm_per_pixel = 1000 / int(args[4])
    plt.hist(np.array(particles) * nm_per_pixel)
    plt.title('Large feature size distribution')
    plt.xlabel('Average Diameter (nm)')
    plt.ylabel('Frequency')
    plt.savefig(args[1] + 'large_dist.tif', bbox_inches='tight')

    small_dmap = np.load(args[1] + 'small_features.npy')
    for cx in range(small_dmap.shape[0]):
        for cy in range(small_dmap.shape[1]):
            if small_dmap[cx, cy]:
                dmap[cx, cy] = small_dmap[cx, cy]
    dm = dmap * nm_per_pixel
    flattened = dm.flatten()
    flattened = flattened[flattened != 0]
    plt.hist(flattened, bins=[i for i in range(0, 1501, 50)])
    plt.title('Size distribution by pixels in features')
    plt.xlabel('Average Diameter (nm)')
    plt.ylabel('Frequency (pixels)')
    plt.savefig(args[1] + 'size_dist.tif', bbox_inches='tight')


if __name__ == '__main__':
    main(sys.argv)
