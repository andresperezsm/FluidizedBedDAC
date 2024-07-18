import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
import scipy.optimize as sopt
import pymrm as mrm

from particle_model import *

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
###############################################################
###############################################################
####                                                       ####
####     ############  ####    ####      ###               ####
####     ############  ####    ####     ###  ######        ####
####         ####      ####    ####    ###  ###  ###       ####
####         ####      ####    ####   ###  ##########      ####
####         ####      ############  ###    ###            ####
####         ####       ##########  ###      ######        ####
####                                                       ####
###############################################################
###############################################################
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def calculate_correlation_parameters():
    T_surface_list = np.linspace(700,1200,6)
    d_p_list = np.logspace(-3,-2,10)
    X_CO2_list = np.linspace(0.1,0.9,9)
    Nodes = 40
    # dp = 4e-3

    T_surface = 1100
    P_surface = 5*101325

    X_CO2_surface = 0.5
    X_H2_surface = 1 - X_CO2_surface

    C_CO2_surface = X_CO2_surface*P_surface/T_surface/8.314
    C_H2_surface = X_H2_surface*P_surface/T_surface/8.314

    c_surface = np.array([C_CO2_surface,C_H2_surface,0,0])

    Q_Joule = 0
    P_surface = 5*101325
    I_current = 0

    results = {T_surface: {'efficiencies': [], 'Thiele moduli': [], 'CO2/H2 ratios': []} for T_surface in T_surface_list}

    for T_surface in T_surface_list:
        for d_p in d_p_list:
            for X_CO2_surface in X_CO2_list:
                
                X_H2_surface = 1 - X_CO2_surface
                
                C_CO2_surface = X_CO2_surface*P_surface/T_surface/8.314
                C_H2_surface = X_H2_surface*P_surface/T_surface/8.314
                
                c_surface = np.array([C_CO2_surface,C_H2_surface,0,0])
                
                particle = particle_model(Nodes,d_p,0.01,c_surface,T_surface,P_surface,Q_Joule,c_dependent_diffusion=True)
                efficiency, Thiele_modulus, r_app_CO2 = particle.solve(1)
                
                CO2_H2_ratio = X_CO2_surface / (1 - X_CO2_surface)
                
                results[T_surface]['efficiencies'].append(efficiency)
                results[T_surface]['Thiele moduli'].append(Thiele_modulus)
                results[T_surface]['CO2/H2 ratios'].append(CO2_H2_ratio)

    def analytical_solution(phi, a, b, c):
        eta_ana = (1 + a*phi**b)**-c
        return eta_ana


    def objective(params, results):
        a, b, c, T_ref, n, k = params
        mse = 0
        count = 0

        for T_surface in results.keys():
            for i, Thiele_modulus in enumerate(results[T_surface]['Thiele moduli']):
                eff_actual = results[T_surface]['efficiencies'][i]
                if abs(eff_actual - 1) > 0.40:  # Check if efficiency is not close to 1
                    CO2_H2_ratio = results[T_surface]['CO2/H2 ratios'][i]
                    modified_thiele = Thiele_modulus * (T_surface / T_ref) ** n * (CO2_H2_ratio) ** k
                    eff_pred = analytical_solution(modified_thiele, a, b, c)
                    mse += (eff_pred - eff_actual) ** 2
                    count += 1

        mse /= count
        return mse

    def fit_modified_thiele_modulus(results):
        initial_guess = [1.0, 1.0, 1.0, 800, 1.7, 1.0]  # Initial guess for a, b, c, T_ref, n, and k
        
        result = sopt.minimize(objective, initial_guess, args=(results), method='Nelder-Mead')
        
        a_optimal, b_optimal, c_optimal, T_ref_optimal, n_optimal, k_optimal = result.x
        
        return a_optimal, b_optimal, c_optimal, T_ref_optimal, n_optimal, k_optimal

    # Fit the modified Thiele modulus
    a_optimal, b_optimal, c_optimal, T_ref_optimal, n_optimal, k_optimal = fit_modified_thiele_modulus(results)

    if __name__ == "__main__":
        print()
        print('Optimal parameters for the modified Thiele modulus')
        print(f"Optimal T_ref: {T_ref_optimal:.2f} K, Optimal n: {n_optimal:.2f}, Optimal k: {k_optimal:.2f}")
        print()
        print('Optimal parameters for particle efficiency')
        print(f"Optimal a: {a_optimal:.2f}, Optimal b: {b_optimal:.2f}, Optimal c: {c_optimal:.2f}")

        # Calculate and plot results with the optimized parameters
        plt.figure(figsize=(10, 6))
        for T_surface in T_surface_list:
            modified_thiele_list = [
                Thiele_modulus * (T_surface / T_ref_optimal) ** n_optimal * (ratio) ** k_optimal 
                for Thiele_modulus, ratio in zip(results[T_surface]['Thiele moduli'], results[T_surface]['CO2/H2 ratios'])
            ]
            plt.plot(modified_thiele_list, results[T_surface]['efficiencies'], marker='o', label=f'T_in = {T_surface} K')

        # Plot the analytical curve for comparison
        thiele_list = np.logspace(-1, 4, 100)
        eta_analytical = [analytical_solution(thiele, a_optimal, b_optimal, c_optimal) for thiele in thiele_list]
        plt.plot(thiele_list, eta_analytical, 'k--', linewidth=1, label='Developed correlation')

        # Calculate upper and lower bounds for error bars
        eta_analytical_upper = [eta * 1.25 for eta in eta_analytical]
        eta_analytical_lower = [eta * 0.75 for eta in eta_analytical]

        # Plot error bars
        plt.fill_between(thiele_list, eta_analytical_upper, eta_analytical_lower, color='gray', alpha=0.2)

        plt.xlabel('Modified Thiele Modulus')
        plt.ylabel('Efficiency')
        plt.title('Efficiency vs. Modified Thiele Modulus with Optimized Parameters')
        plt.grid(which='major',linestyle='-')
        plt.grid(which='minor',linestyle=':')
        plt.minorticks_on()
        plt.xscale('log')  # Use logarithmic scale for Thiele modulus
        plt.yscale('log')  # Use logarithmic scale for Efficiency
        plt.xlim(1e-1, 2e3)
        plt.ylim(1e-3, 2e0)
        plt.legend()
        plt.show()
    
    return a_optimal, b_optimal, c_optimal, T_ref_optimal, n_optimal, k_optimal

a,b,c,T_ref,k,n = calculate_correlation_parameters()