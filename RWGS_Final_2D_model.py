import numpy as np 
import scipy.sparse as sps 
import pymrm as mrm 
import matplotlib.pyplot as plt 
from IPython.display import clear_output, display

# 2D Model
class final2Dmodel: 
    def __init__ (self):
        ##Reactor constants
        self.v = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.u_g = self.v[0]
        self.L = 3
        self.D = 1
        self.H_r = 42e3 #Heat of reaction at (293 K)
        self.eps_b = 0.4 #Bed porosity
        
        ##Gas constants
        self.Rho_g = 1  #Densty of the gas
        self.eta_g = 4e-5 
        self.Cp_g = 5e3 #Heat capacity gas

        ## Maxwell-Stefan constants
        #Moleculaire weight 
        self.M_CO2 = 44  # 1 
        self.M_H2 = 2    # 2
        self.M_CO = 28   # 3
        self.M_H2O = 48  # 4

        #Diffusion volumes
        self.V_CO2 = 26.7 
        self.V_H2 = 6.12
        self.V_CO = 18.0
        self.V_H2O = 13.1

        #Heat capacities
        self.Cp_CO2 = 1.2e3 
        self.Cp_H2 = 15e3
        self.Cp_CO = 1.1e3
        self.Cp_H2O = 2e3

         ## Calculate the average diffusion coefficient for each molecule
        self.Tin = 700
        self.P = 5*101325 
        self.Dm_CO2, self.Dm_H2, self.Dm_CO, self.Dm_H2O = self.calculate_average_diffusion_coefficients()  #Moleculare diffsion coeffcients
        self.lam_g = 0.03 #Thermal diffusivity (WHICH COMPONENT)
        self.Rg = 8.314  #Gas constant

        ##Catalyst constants
        self.d_p = 1e-4 #Diameter of the particle 
        self.Cp_s = 0.45e3 #Heat capacity gas (air)
        self.R_e = 1e-7 ## Electric conductivity Iron
        self.I = 2500000 # 1000000 # 
        print(self.R_e*self.I)
        self.Q_joule = 8e4 # self.R_e*self.I**2 # Joule heatign term 
        self.Dax_s = 0.0
        self.lam_s = 70 #Thermal diffusivity
        self.a_gs = 6/self.d_p*(1-self.eps_b) #Specific surface area
        self.rho_s = 4580 #Solid density
        
        # Dimensionless numbers:
        self.Re = self.Rho_g*self.u_g*self.d_p/self.eta_g # Reynolds number [-]
        self.Pr = self.Cp_g*self.eta_g/self.lam_g # Prandtl number [-]
        self.Sc_CO2 = self.eta_g / (self.Rho_g * self.Dm_CO2) # Schmidt number CO2 [-]
        self.Sc_H2 = self.eta_g / (self.Rho_g * self.Dm_H2) # Schmidt number H2 [-]
        self.Sc_CO = self.eta_g / (self.Rho_g * self.Dm_CO) # Schmidt number CO [-]
        self.Sc_H2O = self.eta_g / (self.Rho_g * self.Dm_H2O) # Schmidt number H2O [-]
        
        # Correlations:
        ## Axial mass dispersion
        self.D_ax_g = np.mean([self.Dm_CO2, self.Dm_H2, self.Dm_CO, self.Dm_H2O])/np.sqrt(2) + 0.5*self.u_g*self.d_p # Dispersion coefficient gas [m2/s]

        ## Axial thermal dispersion
        self.A = self.lam_s/self.lam_g
        self.B = 1.25*((1-self.eps_b)/self.eps_b)**(10/9)
        self.gamma = 2/(1-self.B/self.A)*((self.A-1)/(1-self.B/self.A)**2*self.B/self.A*(np.log(self.A/self.B))-0.5*(self.B+1))
        self.lam_stat = self.lam_g*((1-np.sqrt(1-self.eps_b))+np.sqrt(1-self.eps_b)*self.gamma)
        self.Df_therm = self.lam_g/self.Rho_g/self.Cp_g
        self.Dax_therm = 0.5*self.Re*self.Pr             ##Dispersion coefficient temperature
      
        self.h_w = self.lam_g/self.d_p*(1.3+5/self.D/self.d_p)*self.lam_stat/self.lam_s+0.19*self.Re**(0.75)*self.Pr**(1/3) ##Bed to wall heat transfer 
        
        # Gas solid mass and heat transfer coefficient    
        self.k_gs_CO2 = self.calculate_mass_transfer_coefficient(self.Dm_CO2, self.Sc_CO2)
        self.k_gs_H2 = self.calculate_mass_transfer_coefficient(self.Dm_H2, self.Sc_H2)
        self.k_gs_CO = self.calculate_mass_transfer_coefficient(self.Dm_CO, self.Sc_CO)
        self.k_gs_H2O = self.calculate_mass_transfer_coefficient(self.Dm_H2O, self.Sc_H2O)
        
        self.h_mt = self.lam_g/self.d_p*((7-10*self.eps_b+5*self.eps_b**2)*(1+0.7*self.Re**0.2*self.Pr**0.33)+(1.33-2.4*self.eps_b+1.2*self.eps_b**2)*self.Re**0.7*self.Pr**0.33) # Gunn correlations
        
        self.k_mt = np.array([self.k_gs_CO2, self.k_gs_H2, self.k_gs_CO, self.k_gs_H2O, self.h_mt])

        ## Correlations
        self.Dax_g = np.mean([self.Dm_CO2, self.Dm_H2, self.Dm_CO, self.Dm_H2O]) * (1/np.sqrt(2)) + 0.5*self.v[0]*self.d_p ## Dispersion coefficient gas
        self.Drad_g = np.mean([self.Dm_CO2, self.Dm_H2, self.Dm_CO, self.Dm_H2O]) * (1/np.sqrt(2)) + 0.5*self.v[0]*self.d_p ## Dispersion coefficient gas


        """ ASSUMPTIONS TO BE CHECKED """
        self.Dax_s = 0
        self.Drad_s = self.Drad_g
        self.Drad_therm = 0 #self.Df_therm # 0.5*self.Re*self.Pr             ##Dispersion coefficient temperature


        ## Simulation parameters
        self.nz = 200
        self.nr = 10
        self.nc = 8
        self.nT = 2
        self.dt = np.inf
        self.Tin = 700
        self.Tw = 298
        self.P = 1*101325 
        self.X_H2 = 0.5
        self.X_CO2 = 0.5
        self.Cin_H2 = self.P*self.X_H2/(self.Rg*self.Tin)
        self.Cin_CO2 = self.P*self.X_CO2/(self.Rg*self.Tin)
    
        self.z_f= np.linspace(0, self.L, self.nz +1)
        self.z_c = 0.5*(self.z_f[0:-1]+ self.z_f[1:])

        self.r_f = np.linspace(0, self.D/2, self.nr+1)
        self.r_c = 0.5 * (self.r_f[0:-1] + self.r_f[1:])

        self.c0 = np.array([0.0, 0.0, 0.0, 0.0, self.Tin, 0, 0, 0, 0, self.Tin], dtype='float')
        self.cin = np.array([self.Cin_CO2, self.Cin_H2, 0.0, 0.0, self.Tin, 0.0, 0, 0, 0, 0], dtype='float')

        self.Dax = np.array([self.Dax_g, self.Dax_g, self.Dax_g, self.Dax_g, self.Dax_therm, 0,0,0,0,0], dtype = 'float')
        
        self.bc_ax = {
            'a': [[[self.Dax]], 1],
            'b': [[[self.v]], 0],
            'd': [[[self.v*self.cin]] , 0.0]
        }## C=Cin at Z=0 and zero gradient at Z=L
       

        self.bc_rad = {
            'a': [1, [[[1, 1, 1, 1, 0,       1, 1, 1, 1, 1]]]],
            'b': [0, [[[0, 0, 0, 0, 1,         0, 0, 0, 0, 0]]]],
            'd': [0, [[[0, 0, 0, 0, self.Tw, 0, 0, 0, 0, self.Tw]]]] 
        } ## Boundry condtion wall T

        self.init_field(self.c0)

        self.init_Jac()

        self.freq_out=100

    def calculate_average_diffusion_coefficients(self):
        # Define the molecules and their properties
        molecules = [
            ('CO2', self.M_CO2, self.V_CO2),
            ('H2', self.M_H2, self.V_H2),
            ('CO', self.M_CO, self.V_CO),
            ('H2O', self.M_H2O, self.V_H2O)
        ]
        
        # Calculate diffusion coefficients for each molecule with every other molecule
        diffusion_coeffs = {mol[0]: [] for mol in molecules}
        
        for i, (name1, M1, V1) in enumerate(molecules):
            for j, (name2, M2, V2) in enumerate(molecules):
                if i != j:
                    diffusion_coeff = self.fuller(self.Tin, M1, M2, V1, V2, self.P)
                    diffusion_coeffs[name1].append(diffusion_coeff)
        
        # Calculate average diffusion coefficient for each molecule
        avg_diffusion_coeffs = {name: np.mean(coeffs) for name, coeffs in diffusion_coeffs.items()}
        print(f"Average Diffusion Coefficients: {avg_diffusion_coeffs}")
        
        return avg_diffusion_coeffs['CO2'], avg_diffusion_coeffs['H2'], avg_diffusion_coeffs['CO'], avg_diffusion_coeffs['H2O']
    
    
    def fuller(self, T, M1, M2, V1, V2, P):
        D_ij = 1.013e-2 * T**1.75 / P * np.sqrt(1/M1 + 1/M2) / (V1**(1/3) + V2**(1/3))**2
        return D_ij


    def calculate_mass_transfer_coefficient(self, D_i, Sc_i):
        # Gunn correlation:
        k_mt = D_i/self.d_p*((7 - 10*self.eps_b + 5*self.eps_b**2)*(1 + 0.7*self.Re**0.2 * Sc_i**0.33) + (1.33 - 2.4*self.eps_b + 1.2*self.eps_b**2)*self.Re**0.7*Sc_i**0.33)
        return k_mt

    def init_field(self, c0):
        self.c = np.full([self.nz, self.nr, self.nc+self.nT], c0, dtype='float')
 
    def reaction(self, c): 
        T_p = c[:,:,9]
        T_f = c[:,:,4]

        #Reaction rate constants
        K_H2O = 96808*np.exp(-51979/(self.Rg*T_p))
        k = 11101.2*np.exp(-117432/(self.Rg*T_p))
        Ke = np.exp(-(-12.11+5319*T_p**(-1)+1.012*np.log(T_p)+1.144*10**(-4*T_p)))

        #Pressure calculation
        Pco2 = c[:,:,5]*self.Rg*T_p
        Ph2 = c[:,:,6]*self.Rg*T_p
        Pco = c[:,:,7]*self.Rg*T_p
        Ph2o = c[:,:,8]*self.Rg*T_p

        f = np.zeros_like(c)
        
        #Reaction rate equation
        r = k*(Pco2*Ph2-(Pco*Ph2o)/(Ke))/(Ph2+K_H2O*Ph2o)*self.rho_s*(1-self.eps_b)*1e3/101325

        r[np.isnan(r)]=0
        # r[np.isinf(r)]=0
        # Source/Sinks terms
        f[:,:,0]= -self.k_mt[0]*self.a_gs*(c[:,:,0]-c[:,:,5])   #-r1+r_1
        f[:,:,1] = -self.k_mt[1]*self.a_gs*(c[:,:,1]-c[:,:,6])   # r1-r2-r_1
        f[:,:,2] =  -self.k_mt[2]*self.a_gs*(c[:,:,2]-c[:,:,7])  #r2
        f[:,:,3] =  -self.k_mt[3]*self.a_gs*(c[:,:,3]-c[:,:,8])
        f[:,:,4] =  self.h_mt*self.a_gs*(T_p-T_f)/(self.Rho_g*self.Cp_g)    
        
        f[:,:,5]= self.k_mt[0]*self.a_gs*(c[:,:,0]-c[:,:,5]) -r   #-r1+r_1
        f[:,:,6] = self.k_mt[1]*self.a_gs*(c[:,:,1]-c[:,:,6]) -r   # r1-r2-r_1
        f[:,:,7] =  self.k_mt[2]*self.a_gs*(c[:,:,2]-c[:,:,7]) +r  #r2
        f[:,:,8] =  self.k_mt[3]*self.a_gs*(c[:,:,3]-c[:,:,8]) +r   
        f[:,:,9] =  -self.h_mt*self.a_gs*(T_p-T_f) -self.H_r*r+self.Q_joule*(1-self.eps_b)

        return f
    

    def init_Jac(self):
        self.Jac_accum = sps.diags([1.0, 1.0, 1.0, 1.0, 1.0, (1-self.eps_b), (1-self.eps_b), (1-self.eps_b), (1-self.eps_b), self.rho_s*self.Cp_s*(1-self.eps_b)]*self.nz*self.nr, dtype='float', format='csc')/self.dt
        Grad, grad_bc = mrm.construct_grad(self.c.shape, self.z_f, self.z_c, self.bc_ax, axis=0)
        Conv, conv_bc = mrm.construct_convflux_upwind(self.c.shape, self.z_f, self.z_c, self.bc_ax, self.v, axis=0)
        self.Div_ax = mrm.construct_div(self.c.shape, self.z_f, nu=0, axis=0)
        Dax_m = mrm.construct_coefficient_matrix([[[self.Dax_g, self.Dax_g, self.Dax_g, self.Dax_g, self.Dax_therm, self.Dax_s, self.Dax_s, self.Dax_s, self.Dax_s, 0]]], self.c.shape, axis=0)
        self.Flux = Conv-Dax_m@Grad
        self.flux_bc = conv_bc -Dax_m@grad_bc
        
        Grad, grad_bc = mrm.construct_grad(self.c.shape, self.r_f, self.r_c, self.bc_rad, axis=1)
        Div_rad = mrm.construct_div(self.c.shape, self.r_f, nu=1, axis=1)
        D_rad_m = mrm.construct_coefficient_matrix([[[self.Drad_g, self.Drad_g, self.Drad_g, self.Drad_g, self.Drad_therm, self.Dax_s, self.Dax_s, self.Dax_s, self.Dax_s, 0]]], self.c.shape, axis=1)
        self.Flux_rad = -D_rad_m @ Grad
        self.flux_rad_bc = -D_rad_m @ grad_bc

        self.g_const = self.Div_ax@self.flux_bc + Div_rad @ self.flux_rad_bc
        self.Jac_const = self.Jac_accum + self.Div_ax@self.Flux + Div_rad @ self.Flux_rad

    def lin_pde(self, c, c_old):
        f_react, Jac_react = mrm.numjac_local(self.reaction, c)
        c_f, dc_f = mrm.interp_cntr_to_stagg_tvd(c, self.z_f, self.z_c, self.bc_ax, self.v, mrm.minmod)
        dg_conv = self.Div_ax@(self.v*dc_f).reshape(-1,1)
        g = self.g_const + self.Jac_const@c.reshape(-1,1) + dg_conv - self.Jac_accum@c_old.reshape(-1,1) -f_react.reshape(-1,1)
        Jac = self.Jac_const-Jac_react 
        return g, Jac 
    

    def solve(self, nt): 
        self.plot_pre()
        for i in range(1,nt+1):
            c_old = self.c.copy()
            result = mrm.newton(lambda c: self.lin_pde(c, c_old), c_old)
            self.c = result.x 
            if (i % self.freq_out == 0):
                self.plot()
        self.plot_temperatures()

    def plot_pre(self):
        self.fig, (self.ax1, self.ax2, self.ax3, self.ax4) = plt.subplots(1, 4, figsize=(25, 5))  # Adjust the figsize as needed
        self.contour1 = self.ax1.pcolormesh(self.r_f, self.z_f, self.c[:, :, 0], shading='flat', cmap='viridis', vmin=0, vmax=9)
        self.contour2 = self.ax2.pcolormesh(self.r_f, self.z_f, self.c[:, :, 1], shading='flat', cmap='viridis', vmin=0, vmax=9)
        self.contour3 = self.ax3.pcolormesh(self.r_f, self.z_f, self.c[:, :, 2], shading='flat', cmap='viridis', vmin=0, vmax=9)
        self.contour4 = self.ax4.pcolormesh(self.r_f, self.z_f, self.c[:, :, 3], shading='flat', cmap='viridis', vmin=0, vmax=9)
        self.fig.colorbar(self.contour1, ax=self.ax1)
        self.fig.colorbar(self.contour2, ax=self.ax2)
        self.fig.colorbar(self.contour3, ax=self.ax3)
        self.fig.colorbar(self.contour4, ax=self.ax4)
        self.ax1.set_xlabel('r')
        self.ax1.set_ylabel('z')
        self.ax1.set_title('CO_2')
        self.ax2.set_xlabel('r')
        self.ax2.set_ylabel('z')
        self.ax2.set_title('H_2')
        self.ax3.set_xlabel('r')
        self.ax3.set_ylabel('z')
        self.ax3.set_title('CO')
        self.ax4.set_xlabel('r')
        self.ax4.set_ylabel('z')
        self.ax4.set_title('H_2O')

    def plot(self):
        clear_output(wait=True)
        self.contour1.set_array(self.c[:, :, 0].flatten())
        self.contour2.set_array(self.c[:, :, 1].flatten())
        self.contour3.set_array(self.c[:, :, 2].flatten())
        self.contour4.set_array(self.c[:, :, 3].flatten())
        display(self.fig)
        plt.show()

    def plot_temperatures(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))  # Adjust the figsize as needed
        contour1 = ax1.pcolormesh(self.z_f, self.r_f, self.c[:, :, 4].T, shading='auto', cmap='inferno')
        contour2 = ax2.pcolormesh(self.z_f, self.r_f, self.c[:, :, 9].T, shading='auto', cmap='inferno')
        fig.colorbar(contour1, ax=ax1)
        fig.colorbar(contour2, ax=ax2)
        ax1.set_xlabel('z')
        ax1.set_ylabel('r')
        ax1.set_title('Fluid Temperature (T_f)')
        ax2.set_xlabel('z')
        ax2.set_ylabel('r')
        ax2.set_title('Particle Temperature (T_p)')

        plt.show()