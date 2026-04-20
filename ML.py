from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,ConstantKernel, WhiteKernel
import numpy as np
import joblib
from interp import (
    Interpolator,
    WaterPropertiesInterpolator,
    MgSO4ConstantInterpolator,
    H2SO4ConstantInterpolator,
    H2SO4ConstantInterpolator3D,
    MgSO4SolubilityInterpolator,
    clean_experimental_data_local_outliers,
)


class MgSO4SolubilityML(Interpolator):
    
    def prepare_data(self):
        
        temp_raw = self.df.iloc[:, 0].values
        MgSO4_sol_raw = self.df.iloc[:, 2].values
        H2SO4_conc_raw = self.df.iloc[:, 3].values

        
        mask = clean_experimental_data_local_outliers(
            np.column_stack([temp_raw, H2SO4_conc_raw]), 
            MgSO4_sol_raw,
            return_mask=True,
            z_thresh=10,
            k=5
        )
        self.mask = mask
        # Финальные очищенные данные
        temp = temp_raw[mask]
        MgSO4_sol = MgSO4_sol_raw[mask]
        H2SO4_conc = H2SO4_conc_raw[mask]
        
        # Сохраняем данные для визуализации
        #очищенные точки
        self.points = np.column_stack([temp, H2SO4_conc])
        self.MgSO4_sol = MgSO4_sol
        #исходные точки
        self.points_raw = np.column_stack([temp_raw, H2SO4_conc_raw])
        self.MgSO4_sol_raw = MgSO4_sol_raw

        self.scaler = StandardScaler()
        self.points_normalized = self.scaler.fit_transform(self.points)
        kernel = ConstantKernel(1) * RBF(length_scale=1) + WhiteKernel(noise_level=1e-5)
        self.model = GaussianProcessRegressor(
            kernel = kernel,
            n_restarts_optimizer=5,
            normalize_y=True
        )
        self.model.fit(self.points_normalized, MgSO4_sol)

    def get_sol(self, T_K, H2SO4_conc):
        # Определяем границы доступных значений
        temp_min, H2SO4_conc_min = self.points.min(axis=0)
        temp_max, H2SO4_conc_max = self.points.max(axis=0)
        temp_safe = np.clip(T_K,temp_min, temp_max)
        H2SO4_conc_safe = np.clip(H2SO4_conc,H2SO4_conc_min, H2SO4_conc_max)

        point = np.array([[temp_safe, H2SO4_conc_safe]])
        point_normalized = self.scaler.transform(point)
        mean, std = self.model.predict(point_normalized, return_std=True)
        return float(mean[0]), float(std[0])
