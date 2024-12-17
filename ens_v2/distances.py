from math import radians, sin, cos, sqrt, atan2

def haversine(lon1, lat1, lon2, lat2):
    """
    計算兩個地理坐標之間的哈弗賽距離（公里）。
    """
    R = 6371.0  # 地球半徑，公里

    # 轉換為弧度
    lon1_rad, lat1_rad = radians(lon1), radians(lat1)
    lon2_rad, lat2_rad = radians(lon2), radians(lat2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance