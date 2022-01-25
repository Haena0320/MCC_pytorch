import numpy as np
import matplotlib
from matplotlib import path
matplotlib.use("agg")

def _spaced_points(low, high, n):
    """We want n points between low and high, but we don't wnat them to touch either side"""
    # 가로, 세로로 n 만큼 교차점을 찍을 건데,양쪽 끝은 제외하고 찍고 싶음
    padding = (high-low)/(n*2)
    return np.linspace(low+padding, high-padding, num=n)

# 이미지를 14, 14로 분할했을 때, 해당 오브젝트가 존재하는 범위에 14, 14 교차점이 존재하는지(1), 존재하지 않는지 (0) 체크하는 마스크
def make_mask(mask_size, box, polygons_list):
    mask = np.zeros((mask_size, mask_size), dtype=np.bool)

    xy = np.meshgrid(_spaced_points(box[0], box[2], n=mask_size),
                     _spaced_points(box[1], box[3], n=mask_size))
    # xy : [(14,14), (14,14)]
    xy_flat = np.stack(xy, 2).reshape((-1, 2)) # 교차점의 좌표 리스트 : 14*14 개 [[0.5, 0.5], [0.5, 1], [0.5, 1.5] ...]

    for polygon in polygons_list:
        polygon_path = path.Path(polygon)
        mask |= polygon_path.contains_points(xy_flat).reshape((mask_size, mask_size))
    return mask.astype(np.float32) # [14,14]



