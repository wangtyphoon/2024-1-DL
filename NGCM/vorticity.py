import numpy as np
import xarray as xr

def calculate_vorticity(u_wind, v_wind, latitude, longitude):
    """
    Calculates the relative vorticity using finite differences.

    Args:
        u_wind (xarray.DataArray): Zonal wind component.
        v_wind (xarray.DataArray): Meridional wind component.
        latitude (xarray.DataArray): Latitude coordinates.
        longitude (xarray.DataArray): Longitude coordinates.

    Returns:
        xarray.DataArray: Relative vorticity.
    """

    # 地球半径（米）
    R = 6371000

    # 假设 latitude 和 longitude 是 1D 的 xarray DataArray
    lat_rad = np.deg2rad(latitude)
    lon_rad = np.deg2rad(longitude)

    # 计算纬度和经度方向的间隔（弧度）
    dlat = lat_rad.diff('latitude')
    dlon = lon_rad.diff('longitude')

    # 创建 2D 网格
    lat_rad_2d, lon_rad_2d = xr.broadcast(lat_rad, lon_rad)

    # 计算每个格点处的 dx 和 dy（米）
    dx = R * np.cos(lat_rad_2d) * dlon
    dy = R * dlat

    # 对 u_wind 和 v_wind 进行差分
    du_dy = u_wind.diff('latitude') / dy
    dv_dx = v_wind.diff('longitude') / dx

    # 由于差分操作后维度会减少，需要调整坐标
    du_dy = du_dy.assign_coords(latitude=latitude.isel(latitude=slice(1, None)))
    dv_dx = dv_dx.assign_coords(longitude=longitude.isel(longitude=slice(1, None)))
    
    # 计算涡度 ζ = (∂v/∂x) - (∂u/∂y)
    vorticity = dv_dx - du_dy

    return vorticity
