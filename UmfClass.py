import numpy as np 
import scipy.optimize as opt

class UmfClass:
    def solve_umf(self,d_p):
        # Gas phase properties
        mu_g = 1.825e-5 # Dynamic viscosity of Air at (298 K, 1 bar) (kg m-1 s-1)
        rho_g = 1.204 # Density of air at (293.15 K, 1 atm) (kg m-3)
        g = 9.81 # Gravitational acceleration (m^2/s)

        # Particle phase properties from (Low 2023)
        d_p =  d_p # Lewatit VP OC 1065
        rho_s = 744 # Particle density (kg/m3)
        epsilon_s = 0.338 # Particle porosity (-)
        phi_p = 1 # Particle sphericity assumed to be 1 
        # Archimedes number
        Ar = (g*rho_g*(rho_s - rho_g)*(d_p**3))/((mu_g)**2)

        # Bed voidage at minimum fluidization
        T = 293.15 # Move to reactor properties after
        T0 = 298
        epsilon_mf = 0.382*(((Ar**(-0.196))*((rho_s/rho_g)**(-0.143))) + 1)*((T/T0)**(0.083))

        def minimum_fluidization_vel(Re_mf):
            output = (1.75/((epsilon_mf**3)*phi_p))*(Re_mf**(2)) + ((150*(1 - epsilon_mf))/((epsilon_mf**3)*(phi_p**2)))*(Re_mf**(2)) - Ar
            return output

        initial_guess = 1
        Re_mf = opt.fsolve(minimum_fluidization_vel, initial_guess)
        u_mf = (Re_mf*mu_g)/(d_p*rho_g)
        return Re_mf, u_mf

model = UmfClass()
model.solve_umf(300e-6)