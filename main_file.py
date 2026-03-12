import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import RBFInterpolator
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from chemlib import Compound
from scipy import constants
from scipy.optimize import least_squares

# функция получает молярные массы компонентов, используемых  при расчете
def get_molar_masses():
    return {
        'Fe': Compound("Fe").molar_mass(),
        'S': Compound("S").molar_mass(),
        'As': Compound("As").molar_mass(),
        'K': Compound("K").molar_mass(),
        'NH4': Compound("NH4").molar_mass(),
        'H3O': 19.02,
        'FeS2': Compound("FeS2").molar_mass(),
        'FeAsS': Compound("FeAsS").molar_mass(),
        'Mg(OH)2': Compound("Mg(OH)2").molar_mass(),
        'H2SO4': Compound("H2SO4").molar_mass(),
        'MgSO4': Compound("MgSO4").molar_mass(),
    }

# основные параметры ярозитов
def get_jarosite_params(molar_masses):
    """Получить параметры ярозитов."""
    return {
        'K': {
            'name': 'K-ярозит',
            'dH0': 49683.33,  # Дж/моль
            'dS0': 233.52,    # Дж/(моль·К)
            'M_cation': molar_masses['K'],
            'ion_name': 'K',
            'charge': 1,
            'color': 'blue'
        },
        'H3O': {
            'name': 'H₃O-ярозит',
            'dH0': 90630,
            'dS0': 248.2166667,
            'M_cation': molar_masses['H3O'],
            'ion_name': 'H3O',
            'charge': 1,
            'color': 'red'
        },
        'NH4': {
            'name': 'NH₄-ярозит',
            'dH0': 58920,
            'dS0': 236.4866667,
            'M_cation': molar_masses['NH4'],
            'ion_name': 'NH4',
            'charge': 1,
            'color': 'purple'
        }
    } 


class WaterPropertiesInterpolator:
    """Интерполятор для плотности и диэлектрической проницаемости воды."""
    def __init__(self):
        """Инициализация с табличными данными при 10 МПа."""
        self.table_temps_K = np.array([298.15, 328.15, 373.15, 513.15])
        self.table_epsilon = np.array([88.3, 69.67, 55.4, 30.79])
        self.table_density = np.array([0.72068, 0.65672, 0.55203, 0.27483])
        
        self.epsilon_interp = interp1d(
            self.table_temps_K, self.table_epsilon,
            kind='quadratic', fill_value='extrapolate' 
        )
        self.density_interp = interp1d(
            self.table_temps_K, self.table_density,
            kind='linear', fill_value='extrapolate'
        )
    
    def get_density(self, T_K):
        return float(self.density_interp(T_K))
    
    def get_dielectric(self, T_K):
        return float(self.epsilon_interp(T_K))


class Interpolator:
    
    def __init__(self, file_path, sheet_name):
        self.df = pd.read_excel(file_path, sheet_name=sheet_name)
        self.scaler = None
        self.rbf_interpolator = None
    
    def prepare_data(self):
        raise NotImplementedError

class MgSO4ConstantInterpolator(Interpolator):
    
    def prepare_data(self):
        temp = self.df.iloc[:, 0].values + 273.15
        logK = self.df.iloc[:, 1].values
        K = 10 ** logK
        
        data_df = pd.DataFrame({'temp': temp, 'K': K})
        avg_df = data_df.groupby('temp', as_index=False).mean()
        
        self.temp = avg_df['temp'].values
        self.K = avg_df['K'].values
        
        # Линейная интерполяция
        self.interp_func = interp1d(
            self.temp, self.K,
            kind='quadratic', fill_value='extrapolate'
        )
    
    def get_K(self, T_K):
    
        return float(self.interp_func(T_K))
    
class MgSO4SolubilityInterpolator(Interpolator):
    
    def prepare_data(self):
        temp = self.df.iloc[:, 0].values
        MgSO4_sol = self.df.iloc[:, 1].values
        H2SO4_conc = self.df.iloc[:, 2].values
    
        self.points = np.column_stack([temp, H2SO4_conc])
        self.MgSO4_sol = MgSO4_sol
        
        self.scaler = StandardScaler()
        self.points_normalized = self.scaler.fit_transform(self.points)
        
        self.rbf_interpolator = RBFInterpolator(
            self.points_normalized, MgSO4_sol,
            kernel='linear', smoothing=0.1
        )
    
    def get_sol(self, T_K, H2SO4_conc):
        point = np.array([[T_K, H2SO4_conc]])
        point_normalized = self.scaler.transform(point)
        return float(self.rbf_interpolator(point_normalized)[0])

class H2SO4ConstantInterpolator(Interpolator):
    
    def prepare_data(self):
        temp = self.df.iloc[:, 0].values + 273.15
        logK = self.df.iloc[:, 1].values
        ionic_strength = self.df.iloc[:, 2].values
        
        K = 10 ** logK
        
        self.points = np.column_stack([temp, ionic_strength])
        self.K = K
        
        self.scaler = StandardScaler()
        self.points_normalized = self.scaler.fit_transform(self.points)
        
        self.rbf_interpolator = RBFInterpolator(
            self.points_normalized, K,
            kernel='cubic', smoothing=0.1
        )
    
    def get_K(self, T_K, ionic_strength):
        point = np.array([[T_K, ionic_strength]])
        point_normalized = self.scaler.transform(point)
        return float(self.rbf_interpolator(point_normalized)[0])



def clean_experimental_data_local_outliers(temp, concentration, solubility, 
                                           z_thresh=3, k=15):
    X = np.column_stack([temp, concentration])
    
    nbrs = NearestNeighbors(n_neighbors=k + 1)
    nbrs.fit(X)
    dist, idx = nbrs.kneighbors(X)
    
    mask_good = np.ones(len(solubility), dtype=bool)
    
    for i in range(len(solubility)):
        neigh_idx = idx[i, 1:]  # исключаем саму точку
        local_mean = np.mean(solubility[neigh_idx])
        local_std = np.std(solubility[neigh_idx])
        
        if local_std > 0:
            z = np.abs(solubility[i] - local_mean) / local_std
            if z > z_thresh:
                mask_good[i] = False
    
    return (
        temp[mask_good],
        concentration[mask_good],
        solubility[mask_good]
    )

class CompositionCalculator:

    def __init__(self, molar_masses):
        self.M = molar_masses

    
    def calculate_ore_composition(self, Fe_w, S_w, As_w, K_w, NH4_w,
                                   mass_ore, Ж_Т, Mg_S, Fe_Ox,
                                   H2SO4_add_percent):
        
    
        mass_solid = mass_ore / (Ж_Т + 1)  # г
        mass_liquid = Ж_Т * mass_ore / 1000 * (Ж_Т + 1)  # кг

        # Массы основных компонентов
        m_Fe = Fe_w * mass_solid / 100
        m_S = S_w * mass_solid / 100
        m_As = As_w * mass_solid / 100
        m_K = K_w * mass_solid / 100
        m_NH4 = NH4_w * mass_solid / 100
        
        m_H2SO4_add = H2SO4_add_percent * mass_liquid       
        m_MgOH_add = Mg_S * m_S

        # Расчёт арсенопирита
        n_FeAsS = m_As / self.M['As']
        m_Fe_FeAsS = n_FeAsS * self.M['Fe']
        m_S_FeAsS = n_FeAsS * self.M['S']
        m_Fe_left = m_Fe - m_Fe_FeAsS
        m_S_left = m_S - m_S_FeAsS
        
        # Расчёт пирита
        n_FeS2_from_Fe = m_Fe_left / self.M['Fe']
        n_FeS2_from_S = m_S_left / (2 * self.M['S'])
        n_FeS2 = min(n_FeS2_from_Fe, n_FeS2_from_S)
        
        n_S_excess = 0
        n_Fe_excess = 0
        # Определить избыток
        if n_FeS2_from_Fe < n_FeS2_from_S:
            n_S_excess = (m_S_left / self.M['S']) - (2 * n_FeS2)
            n_Fe_excess = 0
        else:
            n_S_excess = 0
            n_Fe_excess = (m_Fe_left / self.M['Fe']) - (n_FeS2)
    
        self.mass_solid = mass_solid
        self.mass_liquid = mass_liquid
        self.m_Fe = m_Fe
        self.m_S = m_S
        self.m_As = m_As
        self.m_K = m_K
        self.m_NH4 = m_NH4
        self.m_H2SO4_add = m_H2SO4_add
        self.m_MgOH_add = m_MgOH_add
        self.n_FeAsS = n_FeAsS
        self.n_FeS2 = n_FeS2
        self.n_S_excess = n_S_excess
        self.n_Fe_excess = n_Fe_excess

        return self

    
class ActivityCalculator:
    ion_radii = {
        'H': 4.78e-8, 'K': 3.71e-8, 'Na': 4.32e-8,
        'Fe3': 9.0e-8, 'Fe2': 5.08e-8, 'SO4': 5.31e-8,
        'HSO4': 4.5e-8, 'NH4': 2.5e-8, 'Mg2': 8.0e-8,
        'MgSO4': 3.0e-8, 'OH': 4.0e-8
    }
    k_CGS = 1.38054e-16
    e_CGS = 4.80320425e-10
    
    def __init__(self, water_density_func, water_dielectric_func):
        self.get_density = water_density_func
        self.get_epsilon = water_dielectric_func
    
    def get_ion_radius(self, ion_name):
        return self.ion_radii.get(ion_name, 4.0e-8)
    
    def calculate_A(self, T, density, epsilon):
        numerator = (2 * np.pi * constants.N_A)**0.5 * (self.e_CGS**3) * (density**0.5)
        denominator = 2.302585 * 1000**0.5 * (epsilon * self.k_CGS * T)**1.5
        return numerator / denominator
    
    def calculate_B(self, T, density, epsilon):
        numerator = 8 * np.pi * constants.N_A * density * self.e_CGS**2
        denominator = epsilon * self.k_CGS * T * 1000
        return (numerator / denominator)**0.5
    
    def calculate_ionic_strength(self, concentrations, charges):
        return 0.5 * sum(c * z**2 for c, z in zip(concentrations, charges))
    
    def debye_huckel_term(self, z, I, A, B, a):
        sqrt_I = np.sqrt(I)
        denominator = 1 + a * B * sqrt_I
        return -A * z**2 * sqrt_I / denominator
    
    def calculate(self, T, concentrations, charges, ion_names, C_SO4=0):
        density = self.get_density(T)
        epsilon = self.get_epsilon(T)
        
        A = self.calculate_A(T, density, epsilon)
        B = self.calculate_B(T, density, epsilon)
        I = self.calculate_ionic_strength(concentrations, charges)
        
        gammas = []
        dh_terms = []
        si_terms = []
        
        for ion_name, z in zip(ion_names, charges):
            a = self.get_ion_radius(ion_name)
            dh_term = self.debye_huckel_term(z, I, A, B, a)
            si_term = -0.4 * C_SO4
            
            log_gamma = dh_term + si_term
            gamma = 10**log_gamma
            
            gammas.append(gamma)
            dh_terms.append(dh_term)
            si_terms.append(si_term)
        
        return {
            'gamma': gammas,
            'I': I,
            'A': A,
            'B': B,
            'dh_terms': dh_terms,
            'si_terms': si_terms
        }


class SpeciationCalculator:
    
 def __init__(self, T, ion_names, concentrations, charges, 
                 activity_calculator=None, add_Mg=False,
                 h2so4_interp=None, mgso4_interp=None,  # ← передаем интерполяторы
                 max_iter=30, tol=1e-8):
    
        self.T = T
        self.ion_names = ion_names
        self.concentrations = concentrations
        self.charges = charges
        self.add_Mg = add_Mg
        self.max_iter = max_iter
        self.tol = tol
        
        # Интерполяторы
        self.h2so4_interp = h2so4_interp
        self.mgso4_interp = mgso4_interp
        
        # Инициализация
        self.speciation = dict(zip(ion_names, concentrations))
        self.initial_conc = dict(zip(ion_names, concentrations.copy()))
        
        # Балансы
        self.total_H = self.initial_conc.get('H', 0)
        self.total_SO4 = self.initial_conc.get('SO4', 0)
        self.total_Mg = self.initial_conc.get('Mg2', 0) if add_Mg else 0
        
        # История и результаты
        self.I_history = []
        self.gamma = {}
        self.activities = {}
        self.dissociation = {}
        
        # Обработка случая очень малого SO4
        if self.total_SO4 < 1e-5:
            self.speciation["SO4"] = 0
            self.speciation["HSO4"] = 0
            if add_Mg:
                self.speciation["Mg2"] = self.total_Mg
                self.speciation["MgSO4"] = 0
    
 def _get_active_components(self):
    """Получить только компоненты с ненулевой концентрацией"""
    active_ions = []
    active_conc = []
    active_charges = []
        
    for ion, z in zip(self.ion_names, self.charges):
        c = self.speciation.get(ion, 0.0)
        if c > 1e-15:
            active_ions.append(ion)
            active_conc.append(c)
            active_charges.append(z)
        
    return active_ions, active_conc, active_charges
    
 def _update_gamma(self, active_ions, active_conc, active_charges):
       
        result = self.activity_calculator.calculate(
            T=self.T,
            concentrations=active_conc,
            charges=active_charges,
            ion_names=active_ions,
            C_SO4=self.total_SO4
        )
        
        self.gamma = dict(zip(active_ions, result['gamma']))
        self.I = result['I']
        self.A = result['A']
        self.B = result['B']
        self.dh_terms = result['dh_terms']
        self.si_terms = result['si_terms']
        
        return result
    
 def _get_equilibrium_constants(self):
        # Эти функции должны быть доступны глобально или переданы
    K_HSO4 = self.h2so4_interp.get_K(self.T, self.I)
    K_MgSO4 = 0
    if self.add_Mg and self.mgso4_interp is not None:
        K_MgSO4 = self.mgso4_interp.get_K(self.T)
    return K_HSO4, K_MgSO4
    
 def _get_current_concentrations(self):
    H = self.speciation.get("H", 0)
    SO4 = self.speciation.get("SO4", 0)
    HSO4 = self.speciation.get("HSO4", 0)
    Mg = self.speciation.get("Mg2", 0) if self.add_Mg else 0
    MgSO4 = self.speciation.get("MgSO4", 0) if self.add_Mg else 0
    return H, SO4, HSO4, Mg, MgSO4
    
 def _calculate_deltas(self, old_vals, new_vals, names):
    deltas = []
    for old, new, name in zip(old_vals, new_vals, names):
        if old > 0:
            delta = abs(new - old) / old
        else:
            delta = abs(new - old)
        deltas.append(delta)
    return deltas
    
 def _solve_system_no_mg(self, H, SO4, HSO4, γH, γSO4, γHSO4, K_HSO4):
    if self.total_H > 1e-5:
        def eq1(vars):
            H_val, SO4_val, HSO4_val = vars
            return [
                (H_val*γH)*(SO4_val*γSO4) - K_HSO4*(HSO4_val*γHSO4),
                H_val + HSO4_val - self.total_H,
                SO4_val + HSO4_val - self.total_SO4
            ]
        x0 = [H, SO4, HSO4]
        res = least_squares(eq1, x0, bounds=(0, np.inf))
        H_new, SO4_new, HSO4_new = res.x
            
        self.speciation["H"] = H_new
        self.speciation["SO4"] = SO4_new
        self.speciation["HSO4"] = HSO4_new
    else:
        self.speciation["H"] = 0
        self.speciation["HSO4"] = 0
        self.speciation["SO4"] = self.total_SO4
        H_new, SO4_new, HSO4_new = 0, self.total_SO4, 0
        
    return H_new, SO4_new, HSO4_new
    
    def _solve_system_only_mg(self, SO4, Mg, MgSO4, γMg, γSO4, K_MgSO4):
        """Решение системы только с магнием (без H)"""
        def eq2(vars):
            SO4_val, Mg_val, MgSO4_val = vars
            return [
                SO4_val + MgSO4_val - self.total_SO4,
                Mg_val + MgSO4_val - self.total_Mg,
                (Mg_val*γMg)*(SO4_val*γSO4) - K_MgSO4 * MgSO4_val
            ]
        x0 = [SO4, Mg, MgSO4]
        res = least_squares(eq2, x0, bounds=(0, np.inf))
        SO4_new, Mg_new, MgSO4_new = res.x
        
        self.speciation["H"] = 0
        self.speciation["HSO4"] = 0
        self.speciation["SO4"] = SO4_new
        self.speciation["Mg2"] = Mg_new
        self.speciation["MgSO4"] = MgSO4_new
        
        return SO4_new, Mg_new, MgSO4_new
    
    def _solve_system_full(self, H, SO4, HSO4, Mg, MgSO4, 
                           γH, γSO4, γHSO4, γMg, K_HSO4, K_MgSO4):
        """Решение полной системы с H и Mg"""
        def eq3(vars):
            H_val, SO4_val, HSO4_val, Mg_val, MgSO4_val = vars
            return [
                (H_val*γH)*(SO4_val*γSO4) - K_HSO4*(HSO4_val*γHSO4),
                H_val + HSO4_val - self.total_H,
                SO4_val + HSO4_val + MgSO4_val - self.total_SO4,
                Mg_val + MgSO4_val - self.total_Mg,
                (Mg_val*γMg)*(SO4_val*γSO4) - K_MgSO4 * MgSO4_val
            ]
        x0 = [H, SO4, HSO4, Mg, MgSO4]
        res = least_squares(eq3, x0, bounds=(0, np.inf))
        H_new, SO4_new, HSO4_new, Mg_new, MgSO4_new = res.x
        
        self.speciation["H"] = H_new
        self.speciation["SO4"] = SO4_new
        self.speciation["HSO4"] = HSO4_new
        self.speciation["Mg2"] = Mg_new
        self.speciation["MgSO4"] = MgSO4_new
        
        return H_new, SO4_new, HSO4_new, Mg_new, MgSO4_new
    
    def _update_activities(self):
        """Обновить активности"""
        self.activities = {}
        for ion, conc in self.speciation.items():
            if ion in self.gamma:
                self.activities[ion] = conc * self.gamma[ion]
            else:
                self.activities[ion] = conc
    
    def _check_dissociation_constants(self):
        """Проверить константы диссоциации по активностям"""
        if self.activities.get('HSO4', 0) > 1e-15:
            K_calc = (self.activities.get('SO4', 0) * 
                     self.activities.get('H', 0) / 
                     self.activities.get('HSO4', 1))
            self.dissociation['K_HSO4'] = K_calc
            print(f"  K(HSO₄⁻) = {K_calc:.6f}")
        
        if self.add_Mg and self.activities.get('MgSO4', 0) > 1e-15:
            K_calc = (self.activities.get('Mg2', 0) * 
                     self.activities.get('SO4', 0) / 
                     self.activities.get('MgSO4', 1))
            self.dissociation['K_MgSO4'] = K_calc
            print(f"  K(MgSO₄) = {K_calc:.6f}")
    
    def _check_mass_balance(self):
        """Проверить соблюдение материального баланса"""
        final_H_total = self.speciation.get('H', 0) + self.speciation.get('HSO4', 0)
        final_SO4_total = (self.speciation.get('SO4', 0) + 
                          self.speciation.get('HSO4', 0) + 
                          self.speciation.get('MgSO4', 0))
        
        if abs(final_H_total - self.total_H) > 1e-6:
            print(f"\n⚠️  ВНИМАНИЕ: Баланс по водороду нарушен!")
        if abs(final_SO4_total - self.total_SO4) > 1e-6:
            print(f"\n⚠️  ВНИМАНИЕ: Баланс по сере нарушен!")
    
    def calculate(self):
        
        for iteration in range(self.max_iter):
    
            active_ions, active_conc, active_charges = self._get_active_components()
            
            act_result = self._update_gamma(active_ions, active_conc, active_charges)
            
            K_HSO4, K_MgSO4 = self._get_equilibrium_constants()
            
            H, SO4, HSO4, Mg, MgSO4 = self._get_current_concentrations()
            
            γH = self.gamma.get("H", 1.0)
            γSO4 = self.gamma.get("SO4", 1.0)
            γHSO4 = self.gamma.get("HSO4", 1.0)
            γMg = self.gamma.get("Mg2", 1.0)
            
            try:
                if not self.add_Mg:
                    H_new, SO4_new, HSO4_new = self._solve_system_no_mg(
                        H, SO4, HSO4, γH, γSO4, γHSO4, K_HSO4
                    )
                    new_vals = [H_new, SO4_new, HSO4_new]
                    old_vals = [H, SO4, HSO4]
                    names = ['H', 'SO4', 'HSO4']
                    
                else:
                    if self.total_H < 1e-5:
                        SO4_new, Mg_new, MgSO4_new = self._solve_system_only_mg(
                            SO4, Mg, MgSO4, γMg, γSO4, K_MgSO4
                        )
                        new_vals = [SO4_new, Mg_new, MgSO4_new]
                        old_vals = [SO4, Mg, MgSO4]
                        names = ['SO4', 'Mg', 'MgSO4']
                    else:
                        new_vals = self._solve_system_full(
                            H, SO4, HSO4, Mg, MgSO4,
                            γH, γSO4, γHSO4, γMg, K_HSO4, K_MgSO4
                        )
                        old_vals = [H, SO4, HSO4, Mg, MgSO4]
                        names = ['H', 'SO4', 'HSO4', 'Mg', 'MgSO4']
                        
            except Exception as e:
                print(f"⚠️  Ошибка в итерации {iteration}: {e}")
                break
            
            self.I_history.append(self.I)
            
            # 7. Проверка сходимости
            deltas = self._calculate_deltas(old_vals, new_vals, names)
            
            if max(deltas) < self.tol:
                print(f"  ✅ Сходимость за {iteration+1} итераций")
                break
            
            # 8. Обновление активностей и проверка констант
            self._update_activities()
            self._check_dissociation_constants()
            self._check_mass_balance()
        
        # Возврат результатов
        return {
            "speciation": self.speciation,
            "activities": self.activities,
            "gamma": self.gamma,
            "ionic_strength": self.I,
            "iterations": iteration + 1,
            "I_history": self.I_history,
            "dissociation": self.dissociation,
            "A": self.A,
            "B": self.B,
            "density": act_result.get('density') if 'act_result' in locals() else None,
            "epsilon": act_result.get('epsilon') if 'act_result' in locals() else None,
            "dh_terms": self.dh_terms,
            "si_terms": self.si_terms
        }

