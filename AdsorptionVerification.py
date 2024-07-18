import numpy as np 
import matplotlib.pyplot as plt 
import scipy.sparse as sps 
import pymrm  as mrm
import math

class particle_model:
    def __init__(self, Tin, dp):
        ## Gas constants
        self.Rho_g = 1  #Density of the gas
        self.eta_g = 4e-5 #Viscosity gas 
        self.Cp_g = 5e3 #Heat capacity gas
        #self.Dm = 7e-5  #Moleculare diffsion coeffcient
        self.lam_g = 0.03 #Thermal diffusivity
        self.Rg = 8.314  #Gas constant

        ## Maxwell-Stefan constants
        #Molecular weights 
        self.M_CO2 = 44  # 1 
        self.M_N2 = 2    # 2
        self.M_O2 = 32  # 3
        self.M_Ar = 39.95  # 4

        #Diffusion volumes
        self.V_CO2 = 26.7 
        self.V_N2 = 6.12
        self.V_O2 = 16.3 
        self.V_Ar = 16.2

        ## Catalyst constants
        self.d_p = dp #Diameter of the particle 
        self.Cp_s = 0.45e3 #Heat capacity gas (air)
        self.R_e = 1e-7 ## Electric conductivity Iron
        self.I = 1000000 # 2500000 #Current density 
        self.Q_joule = self.R_e*self.I**2 # Joule heating term 
        self.Dax_s = 0.0
        self.lam_s = 70 #Thermal diffusivity
        self.rho_s = 4580 #Solid density
        self.eps_b = 0.4 
        self.eps_s = 0.5 
        self.tauw = np.sqrt(2)
        self.Df_therm = self.lam_g/self.Rho_g/self.Cp_g

        ## Calculate the average diffusion coefficient for each molecule
        self.Tin = 950
        self.P = 1*101325 
        # self.Dm_CO2, self.Dm_H2, self.Dm_CO, self.Dm_H2O = self.calculate_average_diffusion_coefficients(self.Tin, self.P)
        self.Dm_CO2, self.Dm_N2, self.Dm_O2, self.Dm_Ar = self.calculate_average_diffusion_coefficients(self.Tin, self.P)
        
        #Heat of reaction
        self.H_r = 42e3 #Heat of reaction 
        
        self.D_val = 1e-6
        self.k_val = 5 

        #Simulation parameters
        self.Nr = 40
        self.Nc = 1
        self.Tin = Tin
        self.lam_s = 70 

        self.R = dp/2  # Radius of the particle
        self.dp = 2*self.R
        self.dt = np.inf  # Time step (infinity for steady state)
        self.dr_large = 0.1 * self.R  # Initial large grid spacing
        self.refinement_factor = 0.75  # Factor for refining the grid

        self.r_f = mrm.non_uniform_grid(0, self.R, self.Nr + 1, self.dr_large, self.refinement_factor)
        self.r_c = 0.5 * (self.r_f[:-1] + self.r_f[1:])


        self.P = 1*101325 
        self.X_CO2 = 0.04 #The actual figure is 0.0004
        self.X_O2 = 0.2095
        self.X_Ar = 0.0093 
        self.X_N2 = 1 - self.X_CO2 - self.X_O2 #- self.X_Ar
        self.Cin_N2 = self.P*self.X_N2/(self.Rg*self.Tin)
        self.Cin_CO2 = self.P*self.X_CO2/(self.Rg*self.Tin)

        self.c0 = np.array([0,0,0,0, self.Tin,0,0,0,0, self.Tin], dtype='float')


        """ THE FIRST FOUR ARE FOR CO2 IN THE B-CW-E-SOLID PHASE"""
        self.cin = np.array([self.Cin_CO2, self.Cin_CO2, self.Cin_CO2, self.Cin_CO2, self.Tin, 0, 0, 0, 0,  self.Tin], dtype='float')
    
        # Change to a mixed boundary condition the file is in the Analytical solution
        self.bc = {'a':[1,0],'b':[0,1], 'd':[0.0, [self.cin]]}


        # self.P = 1*101325 
        # self.X_CO2 = 0.5
        # self.X_H2 = 1-self.X_CO2
        # self.Cin_H2 = self.P*self.X_H2/(self.Rg*self.Tin)
        # self.Cin_CO2 = self.P*self.X_CO2/(self.Rg*self.Tin)

        # self.c0 = np.array([0,0,0,0, self.Tin], dtype='float')
        # self.cin = np.array([self.Cin_CO2, self.Cin_H2, 0, 0,  self.Tin], dtype='float')
        # self.cin = [1]
        # self.c0 = [0] 
        # self.bc = {'a':[1,0],'b':[0,1], 'd':[0.0, [self.cin]]}

        self.init_field(self.c0)

        self.init_Jac()

        self.freq_out=100

    def fuller(self, T, M1, M2, V1, V2, P):
        D_ij = 1.013e-2 * T**1.75 / P * np.sqrt(1/M1 + 1/M2) / (V1**(1/3) + V2**(1/3))**2
        return D_ij

    def calculate_average_diffusion_coefficients(self, T, P):
        # Define the molecules and their properties
        molecules = [
            ('CO2', self.M_CO2, self.V_CO2),
            ('N2', self.M_N2, self.V_N2),
            ('O2', self.M_O2, self.V_O2),
            ('Ar', self.M_Ar, self.V_Ar)
        ]
        
        # Calculate diffusion coefficients for each molecule with every other molecule
        diffusion_coeffs = {mol[0]: [] for mol in molecules}
        
        for i, (name1, M1, V1) in enumerate(molecules):
            for j, (name2, M2, V2) in enumerate(molecules):
                if i != j:
                    diffusion_coeff = self.fuller(T, M1, M2, V1, V2, P)*self.eps_s/self.tauw
                    diffusion_coeffs[name1].append(diffusion_coeff)
        
        # Calculate average diffusion coefficient for each molecule
        avg_diffusion_coeffs = {name: np.mean(coeffs) for name, coeffs in diffusion_coeffs.items()}
        print(f"Average Diffusion Coefficients: {avg_diffusion_coeffs}")
        
        return avg_diffusion_coeffs['CO2'], avg_diffusion_coeffs['N2'], avg_diffusion_coeffs['O2'], avg_diffusion_coeffs['Ar']


    def init_field(self, c0):
        self.c = np.full([self.Nr, self.Nc], c0, dtype='float')

    ## Reaction rate equations
    def reaction(self, c):
        # T_p = c[:,4]
        T_p = 298 # Temperature of the particle assumed to be 298 K
        # K_H2O = 96808*np.exp(-51979/(self.Rg*T_p))
        # k = 11101.2*np.exp(-117432/(self.Rg*T_p))
        # Ke = np.exp(-(-12.11+5319*T_p**(-1)+1.012*np.log(T_p)+1.144*10**(-4*T_p)))



        pCO2bubble = c[:,0]*self.Rg*T_p
        pCO2cw = c[:,1]*self.Rg*T_p
        pCO2emulsion = c[:,2]*self.Rg*T_p
        pCO2solid = c[:,3]*self.Rg*T_p
        
        # pN2= c[:,1]*self.Rg*T_p
        # pO2 = c[:,2]*self.Rg*T_p
        # pAr = c[:,3]*self.Rg*T_p

        f = np.zeros_like(c)
        
        # r = k*(Pco2-(Pco*Ph2o)/(Ke*Ph2))/(1+K_H2O*Ph2o/Ph2)*self.rho_s*1e3/101325

        # r[np.isnan(r)]=0
        # r = self.k_val*c
        # f = -r


        # f[:,1] = -r 
        # f[:,2] = r
        # f[:,3] = r 
        # f[:,4] = -self.H_r*r/self.rho_s/self.Rho_g+ self.Q_joule/self.rho_s/self.Cp_s   



        return f 

    def init_Jac(self):
        self.Jac_accum= 1/self.dt*sps.eye_array(self.Nr*self.Nc)
        self.Grad, grad_bc = mrm.construct_grad(self.c.shape, self.r_f, self.r_c, self.bc, axis=0)
        self.Div = mrm.construct_div(self.c.shape, self.r_f, nu=2, axis =0)
        # Dax_m = mrm.construct_coefficient_matrix([[self.Dm_CO2, self.Dm_H2, self.Dm_CO, self.Dm_H2O, self.Df_therm]], self.c.shape, axis=0)
        Dax_m =mrm.construct_coefficient_matrix([[self.D_val]], self.c.shape, axis=0)
        self.Flux = -Dax_m@self.Grad
        self.flux_bc = -Dax_m@grad_bc
        self.g_const = self.Div@self.flux_bc
        self.Jac_const = self.Jac_accum + self.Div@self.Flux

    def lin_pde(self, c, c_old):
        f_react, Jac_react = mrm.numjac_local(self.reaction, c)
        g = self.g_const + self.Jac_const@c.reshape(-1,1)  - self.Jac_accum@c_old.reshape(-1,1) -f_react.reshape(-1,1)
        Jac = self.Jac_const-Jac_react 

        return g, Jac 
    
    def solve(self, nt):
        # self.plot_pre()
        for i in range(1, nt + 1):
            c_old = self.c.copy()
            result = mrm.newton(lambda c: self.lin_pde(c, c_old), c_old)
            self.c = result.x
            # if (i% self.freq_out==0):
            #     self.plot(i*self.dt)


        # # Calculate the flux for each component
        # N_CO2 = -self.Dm_CO2 * (self.c[-1, 0] - self.c[-2, 0]) / (self.r_c[-2] - self.r_c[-1])
        # N_H2 = -self.Dm_H2 * (self.c[-1, 1] - self.c[-2, 1]) / (self.r_c[-2] - self.r_c[-1])

        # # Determine the apparent reaction rate for each component
        # rapp_CO2 = 6 * N_CO2 / self.dp
        # rapp_H2 = 6 * N_H2 / self.dp


        # # Determine Weitz-Pater criterion for each component
        # Cwp_CO2 = rapp_CO2 * self.R**2 / self.c[-1, 0] / self.Dm_CO2
        # Cwp_H2 = rapp_H2 * self.R**2 / self.c[-1, 1] / self.Dm_H2

        # print(f"Weitz-Pater criterion for CO2: {Cwp_CO2}")
        # print(f"Weitz-Pater criterion for H2: {Cwp_H2}")

        # T_p = self.c[-1, 4]
        # K_H2O = 96808 * np.exp(-51979 / (self.Rg * T_p))
        # k = 11101.2 * np.exp(-117432 / (self.Rg * T_p))
        # Ke = np.exp(-(-12.11 + 5319 * T_p**(-1) + 1.012 * np.log(T_p) + 1.144 * 10**(-4 * T_p)))

        # Pco2 = self.c[-1, 0] * self.Rg * T_p
        # Ph2 = self.c[-1, 1] * self.Rg * T_p
        # Pco = self.c[-1, 2] * self.Rg * T_p
        # Ph2o = self.c[-1, 3] * self.Rg * T_p

        # # Calculate the rate at the surface of the particle
        # r = k * (Pco2*Ph2 - (Pco * Ph2o) / (Ke)) / (Ph2 + K_H2O * Ph2o ) * self.rho_s * 1e3 / 101325

        # eta = rapp_CO2 / r  # Calculate the efficiency for CO2
        # print(eta)

        N = -self.D_val*(self.c[-1, 0] - self.c[-2, 0]) / (self.r_c[-2] - self.r_c[-1])
        rapp = 6*N/self.dp 

        r = self.k_val*self.c[-1]

        self.thiele = self.d_p / 2 * np.sqrt(r / (self.D_val * self.c[-1, 0]))
        #self.thiele = self.R*np.sqrt(self.k_val/self.D_val)

        eta = rapp/r

        return eta, self.thiele

    def plot_pre(self):
        # plt.ion()
        self.fig1, self.ax1 = plt.subplots(figsize=(10,5))
        self.line1 = []
        labels = ['CO_2', 'H_2', 'CO', 'H_2O']
        for i in range(0, 4):
            self.line1 += self.ax1.plot(self.r_c, self.c[:,i], label=labels[i])
        
        self.ax1.set_title(f'Time: {0.0:5.3f} s', fontsize=16)
        self.ax1.set_xlabel('Position [m]')
        self.ax1.set_ylabel('Concentration [mol/m3]')
        self.ax1.set_xlim(0, self.R)
        self.ax1.set_ylim(0, self.cin[0]*1.2)
        self.ax1.grid()
        self.ax1.legend()


        self.fig2, self.ax2 = plt.subplots(figsize=(10,5))
        self.line2 = []
        
        self.line2 += self.ax2.plot(self.r_c, self.c[:,4], '-', label='T')
        
        self.ax2.set_title(f'Time: {0.0:5.3f} s', fontsize=16)
        self.ax2.set_xlabel('Position [m]')
        self.ax2.set_ylabel('Temperature [k]')
        self.ax2.set_xlim(0, self.R)
        self.ax2.set_ylim()
        self.ax2.grid()
        self.ax2.legend()


    def plot(self, t):
        self.ax1.set_title(f'Time: {t:5.3f} s', fontsize=16)
        for i in range(0, 4): 
            self.line1[i].set_ydata(self.c[:,i])
        self.fig1.canvas.draw()
        self.fig1.canvas.flush_events()

        self.ax2.set_title(f'Time: {t:5.3f} s', fontsize=16)
        self.line2[0].set_ydata(self.c[:,4])    
        self.fig2.canvas.draw()
        self.fig2.canvas.flush_events()    




# particle = particle_model(1100, 1e-3)
# particle.solve(110)
# plt.pause(1e5)

T_in_list = [500, 550, 600, 650, 700, 750, 800, 900, 1000, 1100]
T_in_list = [1300]
dp_list = np.logspace(-5, -1, 20)

print(dp_list)

results = {T_in: {'efficiencies': [], 'thiele_moduli': []} for T_in in T_in_list}

# Loop over the temperatures and particle diameters
for Tin in T_in_list:
    for dp in dp_list:
        particle = particle_model(Tin, dp)
        efficiency, thiele = particle.solve(110)
        results[Tin]['efficiencies'].append(efficiency)
        results[Tin]['thiele_moduli'].append(thiele)

def analytical(thiele):
    eta = 3/thiele*(1/np.tanh(thiele)-1/thiele)

    return eta 

eta_analytical = []

thiele_list = np.logspace(-2, 3, 100)
for thiele in thiele_list:
    eff = analytical(thiele)

    eta_analytical.append(eff) 





# Plot the efficiency vs. Thiele modulus for different temperatures
plt.figure(figsize=(10, 6))
for Tin in T_in_list:
    plt.plot(results[Tin]['thiele_moduli'], results[Tin]['efficiencies'], marker='o', linestyle='-', label=f'T_in = {Tin} K',color = 'orange')

plt.plot(thiele_list, eta_analytical, '--', label='Analytical', color = 'red')

plt.xlabel('Thiele Modulus',fontsize = 16)
plt.ylabel('Efficiency',fontsize = 16)
plt.title('Efficiency vs. Thiele Modulus for Different Inlet Temperatures', fontsize = 20)
plt.grid(True)
plt.xscale('log')  # Use logarithmic scale for Thiele modulus
# plt.xlim(0.01, 18)
plt.legend()
plt.show()