 
from scipy import constants
import numpy as np
import copy
from main_file import (
    SolutionState
)
from scipy.optimize import least_squares, fsolve
import warnings
# расчет активностей ионов 

class ActivityCalculator:
    # константы в сгс
    k_CGS = 1.38054e-16
    e_CGS = 4.80320425e-10

    # берем значения для воды из интерполяционных функций
    def __init__(self, density, epsilon):
        self.get_density = density
        self.get_epsilon = epsilon
    
    # Возвращаем радиус иона из словаря параметров
    def get_ion_radius(self, ion_params):
        return ion_params.get("r", 0)
    
    # расчет параметра А для заданной температуры и параметров воды, (кг^1/2 * моль ^ 1/2)
    def calculate_A(self, T, density, epsilon):
        numerator = (2 * np.pi * constants.N_A)**0.5 * (self.e_CGS**3) * (density**0.5)
        denominator = 2.302585 * 1000**0.5 * (epsilon * self.k_CGS * T)**1.5
        return numerator / denominator
    
    # расчет параметра В для заданной температуры и параметров воды, (кг^1/2 * моль ^ 1/2 / см)
    def calculate_B(self, T, density, epsilon):
        numerator = 8 * np.pi * constants.N_A * density * self.e_CGS**2
        denominator = epsilon * self.k_CGS * T * 1000
        return (numerator / denominator)**0.5

    # расчет ионной силы раствора для всех ионов в растворе
    def calculate_ionic_strength(self, ions):
        I = 0
        for ion_name, params in ions.items():
            I += params["C"] * params["z"]**2
        return 0.5 *I
    
    # уравнение Дебая-Хюккеля
    def debye_huckel_term(self, z, I, A, B, r):
        denominator = 1 + r * B * np.sqrt(I)
        return -A * z**2 * np.sqrt(I) / denominator
    
    def si_term(self, ions, concentrations):

        ε = 0
        for ion_name, params in ions.items():
            if ion_name == 'SO4':
                ε += params.get("ε_SO4", 0) * concentrations.get('SO4', 0)
            elif ion_name == 'HSO4':
                ε += params.get("ε_HSO4", 0) * concentrations.get('HSO4', 0)
        return ε
    
    # объединяем все функции и выполняем расчет логарифма коэффициента активности для данной температуры и набора ионов
    def calculate(self, T, ions, concentrations):
        
        density = self.get_density(T)
        epsilon = self.get_epsilon(T)

        A = self.calculate_A(T, density, epsilon)
        B = self.calculate_B(T, density, epsilon)
        I = self.calculate_ionic_strength(ions)

        for ion_name, params in ions.items():
            z = params["z"] # заряд иона из словаря
            if z == 0:
                params["gamma"] = 1
                params["a"] = params["C"]
            else:
                r = self.get_ion_radius(params) # ионный радиус из словаря
                dh_term = self.debye_huckel_term(z, I, A, B, r)
                si_term = self.si_term(ions, concentrations)

                log_gamma = dh_term + si_term
                gamma = 10**log_gamma
                assert gamma > 0
            
                # обновляем словарь ионов
                params["gamma"] = gamma
                params["a"] = params["C"] * gamma  # активность * концентрация

        # возвращаем обновленный словарь
        return ions, I , si_term


class SpeciationSolver:
    def __init__(self, ions, activity_model, h2so4_interp=None, mgso4_interp=None, add_Mg=False, max_iter=50, tol=1e-8):
        self.activity_model = activity_model
        self.ions = ions
        self.h2so4_interp = h2so4_interp
        self.mgso4_interp = mgso4_interp
        self.add_Mg = add_Mg
        self.max_iter = max_iter
        self.tol = tol

        self.concentrations = {ion: data['C'] for ion, data in ions.items()}
        self.gamma = {ion: 1.0 for ion in ions}
        self.I = 0.0
        self.si_term = 0.0
        self.K_HSO4 = None  # для хранения последней константы HSO4
        self.K_MgSO4 = None  # для хранения последней константы MgSO4
        self.last_cost = 0

        # сохраняем исходные суммы для масс-баланса
        self.total_H = ions['H']['C']
        self.total_SO4 = ions['SO4']['C']
        self.total_Mg = ions.get('Mg', {}).get('C', 0)

    def calculate(self, T, ions=None):
        current_ions = ions if ions is not None else self.ions
        
        for iteration in range(self.max_iter):
        
            # 1. Считаем γ и I при текущих C
            _, I, si_term  = self.activity_model.calculate(T, current_ions, self.concentrations)
            self.I = I
            self.si_term = si_term
            for ion_name in current_ions:
                self.gamma[ion_name] = current_ions[ion_name].get('gamma', 1.0)
        
            # 2. Решаем систему равновесий
            if not self.add_Mg:
               new_conc = self._solve_system_no_mg(T) 
                
            else:
                new_conc = self._solve_system_only_mg(T)
            
            # 4. Проверяем сходимость концентраций
            deltas = [abs(self.concentrations.get(ion, 0) - new_val) 
                for ion, new_val in new_conc.items()]
        
            # 5. Обновляем состояние 
            self.concentrations.update(new_conc)
            for ion, new_val in new_conc.items():
                if ion in current_ions:
                    current_ions[ion]['C'] = new_val

            # 6. Оба критерия сходимости
            converged_c = max(deltas) < self.tol
            converged_r = self.last_cost < 1e-8
        
            if converged_c and converged_r:
                break

            # Проверяем, достигнуто ли максимальное количество итераций
        else:  # Этот блок выполняется, если цикл завершился без break
            warnings.warn(
                f"Достигнуто максимальное количество итераций ({self.max_iter}). "
                f"Сходимость не достигнута: max(delta) = {max(deltas):.2e}, "
                f"cost = {self.last_cost:.2e}",
                UserWarning
            )
        return self
   
    def _solve_system_no_mg(self, T):
        """Решение системы без магния"""
        H = self.concentrations.get("H", 0)
        SO4 = self.concentrations.get("SO4", 0)
        HSO4 = self.concentrations.get("HSO4", 0)
        
        γH = self.gamma.get("H", 1.0)
        γSO4 = self.gamma.get("SO4", 1.0)
        γHSO4 = self.gamma.get("HSO4", 1.0)
        
        K_HSO4 = self.h2so4_interp.get_K(T)
        
        def eq(vars):
            SO4_val, H_val, HSO4_val = vars
            return [
                SO4_val + HSO4_val - self.total_SO4,
                H_val + HSO4_val - self.total_H,
                (SO4_val * γSO4) * (H_val * γH) - K_HSO4 * HSO4_val * γHSO4
            ]
        
        x0 = [SO4, H, HSO4]
        bounds_lo = [0,0,0]
        bounds_hi = [self.total_SO4+1e-10, self.total_H+1e-10, min(self.total_SO4,self.total_H)+1e-10]
        res = least_squares(eq, x0, bounds=(bounds_lo,bounds_hi))
        solution = res.x
        self.last_cost = res.cost       
        SO4_new, H_new, HSO4_new = solution
        self.K_HSO4 = K_HSO4
        return {
            "SO4": SO4_new,
            "H": H_new,
            "HSO4": HSO4_new
        }
    
    def _solve_system_only_mg(self, T):
        """Решение системы с магнием"""
        H = self.concentrations.get("H", 0)
        SO4 = self.concentrations.get("SO4", 0)
        Mg = self.concentrations.get("Mg", 0)
        MgSO4 = self.concentrations.get("MgSO4", 0)
        HSO4 = self.concentrations.get("HSO4", 0)
        
        γSO4 = self.gamma.get("SO4", 1.0)
        γMg = self.gamma.get("Mg", 1.0)
        γH = self.gamma.get("H", 1.0)
        γHSO4 = self.gamma.get("HSO4", 1.0)
        
        K_MgSO4 = self.mgso4_interp.get_K(T) 
        K_HSO4 = self.h2so4_interp.get_K(T)
        
        def eq(vars):
            SO4_val, H_val, HSO4_val, Mg_val, MgSO4_val = vars
            return [
                (SO4_val * γSO4) * (H_val * γH) - K_HSO4 * HSO4_val * γHSO4,
                H_val + HSO4_val - self.total_H,
                SO4_val + MgSO4_val + HSO4_val - self.total_SO4,
                Mg_val + MgSO4_val - self.total_Mg,
                (Mg_val * γMg) * (SO4_val * γSO4) - K_MgSO4 * MgSO4_val
            ]
        
        x0 = [SO4, H, HSO4, Mg, MgSO4]  
        bounds_lo = [0,0,0,0,0]
        bounds_hi = [self.total_SO4 + 0.1, self.total_H+0.1, min(self.total_SO4,self.total_H)+0.1, self.total_Mg+0.1, min(self.total_SO4,self.total_Mg)+0.1]
        res = least_squares(eq, x0, bounds=(bounds_lo,bounds_hi))
        solution = res.x
        self.last_cost = res.cost  
        
        SO4_new, H_new, HSO4_new, Mg_new, MgSO4_new = solution
        self.K_MgSO4 = K_MgSO4
        self.K_HSO4 = K_HSO4
        return {
            "SO4": SO4_new,
            "H": H_new,
            "HSO4": HSO4_new,
            "Mg": Mg_new,
            "MgSO4": MgSO4_new
        }
     
    def get_equilibrium_constants(self):
        """Возвращает последние значения констант равновесия"""
        return {
            'K_HSO4': self.K_HSO4,
            'K_MgSO4': self.K_MgSO4,
            'I': self.I,
            'si_term': self.si_term
        }