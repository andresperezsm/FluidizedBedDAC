import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
import scipy.optimize as sopt
import pymrm as mrm
import time

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

"""
from particle_correlation import *

T_bulk = 1200
P_bulk = 5*101325

X_CO2_bulk = 0.5
X_H2_bulk = 1 - X_CO2_bulk

C_CO2_bulk = X_CO2_bulk*P_bulk/T_bulk/8.314
C_H2_bulk = X_H2_bulk*P_bulk/T_bulk/8.314

c_bulk = np.array([C_CO2_bulk,C_H2_bulk,0,0])

dp = 1e-2

I_current = 0

part = particle_model(dp,c_bulk,T_bulk,P_bulk,I_current)
part.solve(1)

a,b,c,T_ref,k,n = calculate_correlation_parameters()

print(a,b,c,T_ref,k,n) 
"""

class reactor_model: 
    def __init__(self,Nodes, d_p, Lr, dt, gas_velocity, X_CO2_in, T_in, P_in, Q_Joule ,Maxwell_Stefan=False,c_dependent_diffusion=False):
        
        self.Maxwell_Stefan = Maxwell_Stefan
        self.c_dependent_diffusion = c_dependent_diffusion

        # Simulation and field parameters
        self.Nz = Nodes # Number of grid cells in the axial direction [-]
        self.dt = dt # Lengt of a time step [-]
        
        self.Nc = 5 # Number of components (temperature included) [-]
        self.Nph = 2 # Number of phases
        
        self.L_R = Lr # Reactor length [m]
        self.D_R = 1.0 # Reactor diameter [m]
        
        self.t_end = 1.5*self.L_R/gas_velocity
        
        if dt == np.inf:
            self.Nt = 1
        else:
            self.Nt = int(self.t_end/self.dt)
        
        self.dz = self.L_R/self.Nz # Width of a grid cell [m]
        
        self.z_f = np.linspace(0,self.L_R,self.Nz+1) # Axial coordinates of the faces of the grid cells
        self.z_c = 0.5 * (self.z_f[1:] + self.z_f[:-1]) # Axial coordinates of the centers of the grid cells

        # Gas constants
        self.rho_g = 1.225 # Density of air [kg/m3]
        self.eta_g = 4e-5 # Viscosity of air [Pas s]
        self.Cp_g = 5e3 # Molar heat capacity of air [J/kg/K]
        self.lam_g = 0.03 # Thermal conductivity of air [W/m/K]
        self.R_gas = 8.314 # Gas constant [J/mol/K]
        self.P_in = P_in # Reactor pressure [Pa]
        self.eps_b = 0.4 # Bed porosity [-]
        self.u_g = gas_velocity # Velocity of the gas [m/s]
        self.Cp_CO2 = 1.2e3 # Heat capacity of CO2 [J/kg/K]
        self.Cp_H2 = 15e3 # Heat capacity of H2 [J/kg/K]
        self.Cp_CO = 1.1e3 # Heat capacity of CO [J/kg/K]
        self.Cp_H2O = 2e3 # Heat capcity of water [J/kg/K]
        self.Cp_g = 0.5*(self.Cp_CO2+self.Cp_H2) # Mean heat capcity of the feed gas [J/kg/K]
        
        # Catalyst parameters:
        self.d_p = d_p # Diameter of the catalyst particle [m]
        self.Cp_s = 451 # Molar heat capacity of iron [J/kg/K]
        self.D_ax_s = 0.0 # Radial dispersion coefficient in the particle [m2/s]
        self.lam_s = 73 # Thermal conductivity of iron [W/m/K]
        self.rho_s = 4580 # Density of iron [kg/m3]
        self.eps_s = 0.5 # Catalyst porosity [-]
        self.tau = np.sqrt(2) # Catalyst tortuosity [-]
        self.a_gs = 6/self.d_p*(1-self.eps_b) # Specific surface area [m2p/m3r]
        
        # Reaction related parameters:
        self.H_r = 42e3 # Heat of the reaction in [J/mol]
        self.T_in = T_in # Inlet temperature [K]
        
        # Maxwell-Stefan related parameters:
        self.M_CO2 = 44 # Molecular weight of CO2 [g/mol]
        self.M_H2 = 2 # Molecular weight of H2 [g/mol]
        self.M_CO = 28 # Molecular weight of CO [g/mol]
        self.M_H2O = 48 # Molecular weight of H2O [g/mol]

        self.V_CO2 = 26.7 # Diffusion volume of CO2 [m3]
        self.V_H2 = 6.12 # Diffusion volume of H2 [m3]
        self.V_CO = 18.0 # Diffusion volume of CO [m3]
        self.V_H2O = 13.1 # Diffusion volume of H2O [m3]
        
        self.D_CO2, self.D_H2, self.D_CO, self.D_H2O = self.calculate_average_diffusion_coefficients() # Diffusion coefficients [m2/s]
        self.D_eff_p_avg = np.array(self.calculate_average_diffusion_coefficients()) # Effective diffusion coefficients [m2/s]
        
        self.km_12, self.km_13, self.km_14, self.km_23, self.km_24, self.km_34 = self.k_ms()
        
        # Joule heating related parameters:
        # self.I_current = I_current # Current density [A/m2]
        # self.R_ohm = 1e-7 # Ohmic resistance of iron [Ohm/m]
        # self.Q_joule = self.R_ohm*self.I_current**2 # Joule heating term [W]
        self.Q_joule = Q_Joule # Direct joule heating term [W]
        
        # Dimensionless numbers:
        self.Re = self.rho_g*self.u_g*self.d_p/self.eta_g # Reynolds number [-]
        self.Pr = self.Cp_g*self.eta_g/self.lam_g # Prandtl number [-]
        self.Sc_CO2 = self.eta_g / (self.rho_g * self.D_CO2) # Schmidt number CO2 [-]
        self.Sc_H2 = self.eta_g / (self.rho_g * self.D_H2) # Schmidt number H2 [-]
        self.Sc_CO = self.eta_g / (self.rho_g * self.D_CO) # Schmidt number CO [-]
        self.Sc_H2O = self.eta_g / (self.rho_g * self.D_H2O) # Schmidt number H2O [-]
        
        # Correlations:
        self.D_ax_g = np.mean([self.D_CO2, self.D_H2, self.D_CO, self.D_H2O])/np.sqrt(2) + 0.5*self.u_g*self.d_p # Dispersion coefficient gas [m2/s]

        self.A = self.lam_s/self.lam_g 
        self.B = 1.25*((1-self.eps_b)/self.eps_b)**(10/9)
        self.gamma = 2/(1-self.B/self.A)*((self.A-1)/(1-self.B/self.A)**2*self.B/self.A*(np.log(self.A/self.B))-0.5*(self.B+1))
        self.lam_stat = self.lam_g*((1-np.sqrt(1-self.eps_b))+np.sqrt(1-self.eps_b)*self.gamma)
        
        self.D_Thermal = self.lam_g/self.rho_g/self.Cp_g # Thermal diffusivity of air [m2/s]
        self.Dax_Thermal = self.D_Thermal*0.5*self.Re*self.Pr # Axial dispersion coefficient temperature [m2/s]
        self.h_w = self.lam_g/self.d_p*(1.3+5/self.D_R/self.d_p)*self.lam_stat/self.lam_s + 0.19*self.Re**(0.75)*self.Pr**(1/3) # Bed to wall heat transfer [?]
        
        # Gas solid mass and heat transfer coefficient    
        self.k_gs_CO2 = self.calculate_mass_transfer_coefficient(self.D_CO2, self.Sc_CO2)
        self.k_gs_H2 = self.calculate_mass_transfer_coefficient(self.D_H2, self.Sc_H2)
        self.k_gs_CO = self.calculate_mass_transfer_coefficient(self.D_CO, self.Sc_CO)
        self.k_gs_H2O = self.calculate_mass_transfer_coefficient(self.D_H2O, self.Sc_H2O)
        
        self.h_gs = self.lam_g/self.d_p*((7-10*self.eps_b+5*self.eps_b**2)*(1+0.7*self.Re**0.2*self.Pr**0.33)+(1.33-2.4*self.eps_b+1.2*self.eps_b**2)*self.Re**0.7*self.Pr**0.33) # Gunn correlations
        
        self.k_gs = np.array([self.k_gs_CO2, self.k_gs_H2, self.k_gs_CO, self.k_gs_H2O, self.h_gs])
        
        # Inlet conditions:
        self.T_in = T_in # Inlet temperature [K]
        self.P_in = P_in # Inlet pressure [Pa]
        
        self.X_CO2_in = X_CO2_in # Bulk CO2 mole fraction [-]
        self.X_H2_in = 1 - 0.5 # Bulk H2 mole fraction [-]
        
        self.C_CO2_in = self.P_in*self.X_CO2_in/(self.R_gas*self.T_in) # Bulk CO2 concentration [mol/m3]
        self.C_H2_in = self.P_in*self.X_H2_in/(self.R_gas*self.T_in) # Bulk H2 concentration [mol/m3]
        self.C_tot_in = self.C_CO2_in+self.C_H2_in # Total inlet concentration [mol/m3]
        
        self.vel = np.concatenate([np.full(self.Nc,self.u_g),np.zeros(self.Nc),[1]])

        if dt == np.inf: 
            # Initial guess (steady state model)
            self.c_0 = np.ones(self.Nc*self.Nph+1)*self.C_CO2_in*0.5 
            self.c_0[self.Nc-1], self.c_0[2*self.Nc-1] = self.T_in, self.T_in
            self.c_0[-1] = self.P_in
        else: 
            # Initial conditions (transient model)
            self.c_0 = np.ones(self.Nc*self.Nph+1)*1e-3
            self.c_0[self.Nc-1], self.c_0[2*self.Nc-1] = self.T_in, self.T_in
            self.c_0[-1] = P_in
        
        # Boundary conditions
        self.c_in = np.zeros(self.Nc*self.Nph+1) # Initial conditions field
        
        self.c_in[0], self.c_in[1] = self.C_CO2_in, self.C_H2_in
        self.c_in[self.Nc-1], self.c_in[2*self.Nc-1] = self.T_in, self.T_in
        self.c_in[-1] = self.P_in
        
        self.Dax = np.array([self.D_ax_g, self.D_ax_g, self.D_ax_g, self.D_ax_g, self.Dax_Thermal, 0, 0, 0, 0, 0, 0], dtype = 'float')
       
        self.bc_ax = {
                    'a': [[[self.Dax]], 1], # Dirichlet boundary conditions
                    'b': [[[self.vel]], 0], # Neumann boundary conditions
                    'd': [[[self.vel*self.c_in]] , 0.0], # Values
                     }

        # Functions
        self.init_field() # Calls the function in the initial call
        self.init_Jac() # Calls the function in the initial call


    def Fuller_correlation(self, M_i, M_j, V_i, V_j):
        C = 1.013e-2
        D_ij = C*self.T_in**1.75/self.P_in * np.sqrt(1/M_i + 1/M_j) / (V_i**(1/3) + V_j**(1/3))**2
        return D_ij
    
    
    def k_ms(self, correlation='gunn'):
        pairs = [(self.M_CO2, self.M_H2, self.V_CO2, self.V_H2),    # 12
                    (self.M_CO2, self.M_CO, self.V_CO2, self.V_CO),    # 13
                    (self.M_CO2, self.M_H2O, self.V_CO2, self.V_H2O),  # 14
                    (self.M_H2, self.M_CO, self.V_H2, self.V_CO),      # 23
                    (self.M_H2, self.M_H2O, self.V_H2, self.V_H2O),    # 24
                    (self.M_CO, self.M_H2O, self.V_CO, self.V_H2O)]    # 34

        k_ms = []
        
        for M_i, M_j, V_i, V_j in pairs:
            D_ij = self.Fuller_correlation(M_i, M_j, V_i, V_j)

            # Calculate Reynolds number
            Re = self.rho_g * self.u_g * self.d_p / self.eta_g

            # Calculate Schmidt number
            Sc = self.eta_g / (self.rho_g * D_ij)

            if correlation == 'ranz-marshall':
                k_mt = D_ij / self.d_p * (2 + 0.06 * Re**0.5 * Sc**(1/3))
            elif correlation == 'gunn':
                k_mt = D_ij / self.d_p * ((7 - 10 * self.eps_b + 5 * self.eps_b**2) * (1 + 0.7 * Re**0.2 * Sc**0.33) + (1.33 - 2.4 * self.eps_b + 1.2 * self.eps_b**2) * Re**0.7 * Sc**0.33)
            else:
                raise ValueError("Use 'ranz-marshall' or 'gunn'.")

            k_ms.append(k_mt)

        return k_ms
    
    
    def calculate_average_diffusion_coefficients(self):
        diffusion_coeff = np.zeros((self.Nc-1,self.Nc-1))
        M = np.array([self.M_CO2,self.M_H2,self.M_CO,self.M_H2O])
        V = np.array([self.V_CO2,self.V_H2,self.V_CO,self.V_H2O])
        
        for i in range(self.Nc-1): # Takes component i
            for j in range(self.Nc-1): # Calculates binary diffusion of i with each other component j
                if i != j:
                    diffusion_coeff[i,j] = self.Fuller_correlation(M[i], M[j], V[i], V[j]) # Calculates Dij

        avg_diffusion_coeff = np.zeros(self.Nc-1)

        for i in range (self.Nc-1):
            avg_diffusion_coeff[i] = np.sum(diffusion_coeff[i,:])/(self.Nc-2)
        
        return avg_diffusion_coeff[0], avg_diffusion_coeff[1], avg_diffusion_coeff[2], avg_diffusion_coeff[3]


    def calculate_mass_transfer_coefficient(self, D_i, Sc_i):
        # Gunn correlation:
        k_mt = D_i/self.d_p*((7 - 10*self.eps_b + 5*self.eps_b**2)*(1 + 0.7*self.Re**0.2 * Sc_i**0.33) + (1.33 - 2.4*self.eps_b + 1.2*self.eps_b**2)*self.Re**0.7*Sc_i**0.33)
        return k_mt


    def init_field(self):
        self.c = np.full([self.Nz, self.Nc*self.Nph+1], self.c_0, dtype='float')
    
    
    def reaction(self, c): 
        f = np.zeros_like(c)

        T_p = c[:,9]
        T_g = c[:,4]

        c_g = c[:,:4]
        c_s_I = c[:,5:-2]
        P = c[:,-1]

        xavg = 0.5*(c_g+c_s_I)/self.C_tot_in
        dc = c_s_I - c_g

        r = []
        
        for i in range(0, self.Nz):
            # Particle model is called, in the reactor model the particle model is in steady state
            part = particle_model(40,self.d_p,np.inf,c_s_I[i,:],T_p[i],P[i],self.Q_joule,c_dependent_diffusion=self.c_dependent_diffusion)
            eta, phi, rapp = part.solve(1)
            r.append(rapp)

        r = np.array(r)*(1-self.eps_b)

        rho_g = self.rho_g
        Cp_g = self.Cp_g
        
        # rho_g = (c[:,0]*self.M_CO2+c[:,1]*self.M_H2+c[:,2]*self.M_CO+c[:,3]*self.M_H2O)*1e-3
        # Cp_g =  (c[:,0]*self.Cp_CO2+c[:,1]*self.Cp_H2+c[:,2]*self.Cp_CO+c[:,3]*self.Cp_H2O)/(c[:,0]+c[:,1]+c[:,2]+c[:,3])
            
        if self.Maxwell_Stefan == True:
            flx = np.zeros((self.Nz, 4))

            flx[:,0] = - r/self.a_gs
            flx[:,1] = - r/self.a_gs
            flx[:,2] = + r/self.a_gs
            flx[:,3] = + r/self.a_gs 

            f[:,0] = flx[:,0]*self.a_gs
            f[:,1] = flx[:,1]*self.a_gs
            f[:,2] = flx[:,2]*self.a_gs
            f[:,3] = flx[:,3]*self.a_gs

            f[:,4] =  self.h_gs*self.a_gs*(T_p-T_g)/(rho_g*Cp_g)    
            
            f[:,5] = dc[:,0] + (xavg[:,0]*flx[:,1] - xavg[:,1]*flx[:,0])/self.km_12 \
                             + (xavg[:,0]*flx[:,2] - xavg[:,2]*flx[:,0])/self.km_13 \
                             + (xavg[:,0]*flx[:,3] - xavg[:,3]*flx[:,0])/self.km_14
            
            f[:,6] = dc[:,1] + (xavg[:,1]*flx[:,0] - xavg[:,0]*flx[:,1])/self.km_12 \
                            + (xavg[:,1]*flx[:,2] - xavg[:,2]*flx[:,1])/self.km_23 \
                            + (xavg[:,1]*flx[:,3] - xavg[:,3]*flx[:,1])/self.km_24 

            f[:,7] = dc[:,2] + (xavg[:,2]*flx[:,0] - xavg[:,0]*flx[:,2])/self.km_13 \
                            + (xavg[:,2]*flx[:,1] - xavg[:,1]*flx[:,2])/self.km_23 \
                            + (xavg[:,2]*flx[:,3] - xavg[:,3]*flx[:,2])/self.km_34 
            
            f[:,8] = dc[:,3] + (xavg[:,3]*flx[:,0] - xavg[:,0]*flx[:,3])/self.km_14 \
                            + (xavg[:,3]*flx[:,1] - xavg[:,1]*flx[:,3])/self.km_24 \
                            + (xavg[:,3]*flx[:,2] - xavg[:,2]*flx[:,3])/self.km_34 

            f[:,9] =  -self.h_gs*self.a_gs*(T_p-T_g) -self.H_r*r+self.Q_joule*(1-self.eps_b)
           
            # Ergun equation 
            f[:,10] = -(150*self.eta_g*self.u_g*(1-self.eps_b)**2/(self.d_p**2*self.eps_b**3)+1.75*self.rho_g*(1-self.eps_b)/(self.d_p*self.eps_b**2)*self.u_g**2)

        else:
            # Gas phase:
            f[:,0] = -self.k_gs_CO2*self.a_gs*(c[:,0]-c[:,5])
            f[:,1] = -self.k_gs_H2*self.a_gs*(c[:,1]-c[:,6])
            f[:,2] = -self.k_gs_CO*self.a_gs*(c[:,2]-c[:,7])
            f[:,3] = -self.k_gs_H2O*self.a_gs*(c[:,3]-c[:,8])
            f[:,4] = self.h_gs*self.a_gs*(T_p-T_g)/(rho_g*Cp_g)
            
            # Solid phase:
            f[:,5] = self.k_gs_CO2*self.a_gs*(c[:,0]-c[:,5]) - r
            f[:,6] = self.k_gs_H2*self.a_gs*(c[:,1]-c[:,6]) - r
            f[:,7] = self.k_gs_CO*self.a_gs*(c[:,2]-c[:,7]) + r
            f[:,8] = self.k_gs_H2O*self.a_gs*(c[:,3]-c[:,8]) + r   
            f[:,9] = -self.h_gs*self.a_gs*(T_p-T_g) -self.H_r*r + self.Q_joule*(1-self.eps_b)

            # Pressure with Ergun equation:
            f[:,10] = -(150*self.eta_g*self.u_g*(1-self.eps_b)**2/(self.d_p**2*self.eps_b**3)+1.75*self.rho_g*(1-self.eps_b)/(self.d_p*self.eps_b**2)*self.u_g**2)

        return f
    

    def init_Jac(self):
        if self.Maxwell_Stefan == True:
            self.Jac_accum = sps.diags([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]*self.Nz, dtype='float', format='csc')/self.dt
        else: 
            self.Jac_accum = sps.diags([1.0, 1.0, 1.0, 1.0, 1.0, (1-self.eps_b), (1-self.eps_b), (1-self.eps_b), (1-self.eps_b), self.rho_s*self.Cp_s*(1-self.eps_b), 1.0]*self.Nz, dtype='float', format='csc')/self.dt
            
        Grad, grad_bc = mrm.construct_grad(self.c.shape, self.z_f, self.z_c, self.bc_ax, axis=0)
        Conv, conv_bc = mrm.construct_convflux_upwind(self.c.shape, self.z_f, self.z_c, self.bc_ax, self.vel, axis=0)
        self.Div_ax = mrm.construct_div(self.c.shape, self.z_f, nu=0, axis=0)
        Dax_m = mrm.construct_coefficient_matrix([[self.D_ax_g, self.D_ax_g, self.D_ax_g, self.D_ax_g, self.Dax_Thermal, self.D_ax_s, self.D_ax_s, self.D_ax_s, self.D_ax_s, 0, 0.0]], self.c.shape, axis=0)
        self.Flux = Conv-Dax_m@Grad
        self.flux_bc = conv_bc -Dax_m@grad_bc
        self.g_const = self.Div_ax@self.flux_bc
        self.Jac_const = self.Jac_accum + self.Div_ax@self.Flux


    def lin_pde(self, c, c_old):
        f_react, Jac_react = mrm.numjac_local(self.reaction, c)
        c_f, dc_f = mrm.interp_cntr_to_stagg_tvd(c, self.z_f, self.z_c, self.bc_ax, self.vel, mrm.minmod)
        dg_conv = self.Div_ax@(self.vel*dc_f).reshape(-1,1)
        g = self.g_const + self.Jac_const@c.reshape(-1,1) + dg_conv - self.Jac_accum@c_old.reshape(-1,1) - f_react.reshape(-1,1)
        Jac = self.Jac_const-Jac_react 
        return g, Jac 
    

    def solve(self): 
        if __name__ == "__main__":
            plt.figure(figsize=(12,7.5)) 
        
        t = 0
        
        for _ in range(self.Nt):
            # start = time.time()
            c_old = self.c.copy()
            self.c  = mrm.newton(lambda c: self.lin_pde(c, c_old), c_old, tol = 1e-4, maxfev=500).x
            t += self.dt
            
            if __name__ == "__main__":
                
                if self.dt == np.inf: # Steady state plot
                    plt.suptitle('Steady state')
                    
                    plt.subplot(131)
                    labels = ['CO2','H2','CO','H2O']
                    for i in range(self.Nc-1):
                        plt.plot(self.z_c,self.c[:,i],label=labels[i]+' Gas')
                        plt.plot(self.z_c,self.c[:,i+self.Nc],'--',label=labels[i]+' Solid')
                    plt.legend()
                    
                    plt.subplot(132)
                    plt.plot(self.z_c,self.c[:,4],label='Gas temperature')
                    plt.plot(self.z_c,self.c[:,9],'--',label='Solid temperature')
                    plt.legend()
                        
                    plt.subplot(133)
                    plt.plot(self.z_c,self.c[:,10],label='Pressure')
                    plt.legend()
                    # print(f'Time to solve time step: {time.time()-start:.2f} s')
                    plt.show()
                    
                # else: # Transient plot
                #     plt.clf()
                #     plt.suptitle(f'Time: {t:.2f} s')
                #     plt.subplot(131)
                #     labels = ['CO2','H2','CO','H2O']
                #     for i in range(self.Nc-1):
                #         plt.plot(self.z_c,self.c[:,i],label=labels[i]+' Gas')
                #         plt.plot(self.z_c,self.c[:,i+self.Nc],'--',label=labels[i]+' Solid')
                #     plt.legend()
                    
                #     plt.subplot(132)
                #     plt.plot(self.z_c,self.c[:,4],label='Gas temperature')
                #     plt.plot(self.z_c,self.c[:,9],'--',label='Solid temperature')
                #     plt.legend()
                        
                #     plt.subplot(133)
                #     plt.plot(self.z_c,self.c[:,10],label='Pressure')
                #     plt.legend()
                #     plt.pause(0.01)
                #     print(f'Time to solve time step: {time.time()-start:.2f} s')
            
            self.Conversion_CO2 = 1 - self.c[-1,2]/(self.C_CO2_in)      
            # print(f'Conversion CO2 = {self.Conversion_CO2*100:.1f}%')
            self.pdrop = self.c[0,10] - self.c[-1,10]     
            # print(f'Pressure drop = {self.pdrop:.1f} bar')

            return self.Conversion_CO2, self.pdrop
            
if __name__ == "__main__":
    # Steady state:
    reactor = reactor_model(100, np.inf, 3.0, 0.35, 1000, 5*101325, 0, Maxwell_Stefan=True, c_dependent_diffusion=True)
    reactor.solve()
    
    # # Transient:
    # reactor = reactor_model(10, 0.1, 1.0, 0.5, 1100, 5*101325, 0, Maxwell_Stefan=False, c_dependent_diffusion=False)
    # reactor.solve()