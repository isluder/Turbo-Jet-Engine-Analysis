import math
import numpy as np
import pandas as pd


class turbojet_engine:
    def __init__(self):
        #Air - cold-air standard analysis
        self.R = 287                        # gas constant, J/kg-K
        self.k = 1.4                        # specific heat ratio, assumed for ideal air
        self.cp = self.k*self.R/(self.k-1)  #specific heat, J/kg-K
        
        #Input Design Parameters
        
        #Internal axial Mach Number
        self.Max = 0.2
        
        #Inlet Conditions
        self.p1 = 26.2                               #pressure, kPa (for elevation of 10000m)
        self.T1 = 223                                #temperature, K
        self.V1 = 260                                #cruise speed, m/s
        
        
        #Other Design Parameters
        self.rp = 32                                    #pressure ratio in compressor
        self.T4 = 1400                                  #the highest temperature of the air after burner, K
        self.Q_R = 42000*10**3                          #fuel heating value, J/kg
        
        #Device Efficiencies
        self.eta_d = 0.9        #adiabatic efficiency of diffuser
        self.eta_pc = 0.9       #polytropic efficiency of compressor
        self.eta_pt = 0.85      #polytropic efficiency of turbine
        self.eta_n = 0.9        #adaibatic efficiency of nozzle
        
        self.pi_b = 0.95        #pressure ratio in burner
        self.eta_b = 0.992      #combustion efficiency
        
        #Entropy Reference Point
        self.s1 = 100
        self.s01 = self.s1
        
    def calc_Ma1(self):
        self.c1 = np.sqrt(self.k*self.R*self.T1)   #speed of sound, m/s
        self.Ma1 = self.V1/self.c1               #cruise Mach Number

    def state1(self):
            #1) Entering the diffuser (ambient air)        
        self.T01 = self.T1*(1 + ((self.k-1)/2)*(self.Ma1**2))
        self.p01 = self.p1*(self.T01/self.T1)**(self.k/(self.k-1))
        
    def state2(self):
    #2) End of diffuser and inlet to compressor    
        self.T02 = self.T01
        self.Ma2 = self.Max
        self.T2 = self.T02*(1 + ((self.k-1)/2)*self.Ma2**2)**(-1)
        self.p02 = self.p1*(1 + self.eta_d*(self.k-1)/2*self.Ma1**2)**(self.k/(self.k-1))
        self.p2 = self.p02*(self.T2/self.T02)**(self.k/(self.k-1))
        self.s02 = self.s01 + self.cp*np.log(self.T02/self.T01)-self.R*np.log(self.p02/self.p01)
        self.s2 = self.s02
        
        self.V2 = self.Max*np.sqrt(self.k*self.R*self.T2)    
    
    def state3(self):
    #3) End of Compressor and inlet to combustion chamber
        self.p03 = self.p02*self.rp
        self.Ma3 = self.Max
        self.T03 = self.T02*self.rp**((self.k-1)/self.k/self.eta_pc)
        self.T3 = self.T03*(1 + (self.k-1)/2*self.Ma3**2)**(-1)
        self.p3 = self.p03*(self.T3/self.T03)**(self.k/(self.k-1))
        self.s03 = self.s02 + self.cp*np.log(self.T03/self.T02)-self.R*np.log(self.p03/self.p02)
        self.s3 = self.s03
        
        self.eta_c = (1-self.rp**((self.k-1)/self.k))/(1-self.rp**((self.k-1)/self.k/self.eta_pc))
        self.wc = self.cp*(self.T03 - self.T02)          # actual compressor work, kJ/kg
        
        self.V3 = self.Ma3*np.sqrt(self.k*self.R*self.T3)
        
    def state4(self):
    #4) End of combustion chamber and inlet to turbine
        self.p04 = self.pi_b*self.p03
        self.Ma4 = self.Max
        self.T04 = self.T4*(1 + (self.k-1)/2*self.Ma4**2)
        self.p4 = self.p04*(self.T4/self.T04)**(self.k/(self.k-1))
        self.s04 = self.s03 + self.cp*np.log(self.T04/self.T03)-self.R*np.log(self.p04/self.p03)
        self.s4 = self.s04
        #   fuel-to-air ratio
        self.f = (self.T04 - self.T03)/(self.Q_R*self.eta_b/self.cp - self.T04)
        
        self.V4 = self.Ma4*np.sqrt(self.k*self.R*self.T4)
    
    def state5(self):
    #5) End of turbine and inlet to nozzle
    #       all power output from turbine runs the compressor

        self.Ma5 = self.Max
        self.T05 = self.T04 - (self.T03 - self.T02)/(1+self.f)
        self.T5 = self.T05*(1 + (self.k-1)/2*self.Ma5**2)**(-1)
        self.p05 = self.p04*(self.T05/self.T04)**(self.k/(self.k-1)/self.eta_pt)
        self.p5 = self.p05*(self.T5/self.T05)**(self.k/(self.k-1))
        self.s05 = self.s04 + self.cp*np.log(self.T05/self.T04)-self.R*np.log(self.p05/self.p04)
        self.s5 = self.s05
        
        self.pi_t = self.p05/self.p04
        self.eta_t = (1-self.pi_t**((self.k-1)*self.eta_pt/self.k))/(1-self.pi_t**((self.k-1)/self.k))
        
        self.V5 = self.Ma5*np.sqrt(self.k*self.R*self.T5) 
        
    def state6(self):
    #6) End of nozzle with perfect expansion
        self.p6 = self.p1
        self.T6 = self.T05*(1-self.eta_n*(1-(self.p6/self.p05)**((self.k-1)/self.k)))
        self.V6 = np.sqrt(2*self.cp*(self.T05-self.T6))
        self.Ma6 = self.V6/np.sqrt(self.k*self.R*self.T6)
        self.T06 = self.T6*(1 + (self.k-1)/2*self.Ma6**2) 
        self.p06 = self.p6*(self.T06/self.T6)**(self.k/(self.k-1))
        self.s06 = self.s05 + self.cp*np.log(self.T06/self.T05)-self.R*np.log(self.p06/self.p05)
        self.s6 = self.s06       
    
    def cycle_analysis(self):
        self.calc_Ma1()
        
        self.state1()
        
        self.state2()
        
        self.state3()

        self.state4()
        
        self.state5()

        self.state6()

    def perform_analysis(self):
        self.F_m = (1+self.f)*self.V6 - self.V1
        self.TSFC = self.f/((1+self.f)*self.V6-self.V1)*1e6;                  # mg/s/N
        self.eta_th = ((1+self.f)*self.V6**2-self.V1**2)/2/self.f/self.Q_R
        self.eta_PP = 2*(self.V6-self.V1)*self.V1/((1+self.f)*self.V6**2-self.V1**2)
        
    def gather_states(self):
        self.states = pd.DataFrame(data = {'p (kPa)':[self.p1, self.p2, self.p3, self.p4, self.p5, self.p6],
                       'T (K)':[self.T1, self.T2, self.T3, self.T4, self.T5, self.T6],
                       'p0 (kPa)':[self.p01, self.p02, self.p03, self.p04, self.p05, self.p06],
                       'T0 (K)':[self.T01, self.T02, self.T03, self.T04, self.T05, self.T06],
                       's0 (J/Kg-K)':[self.s01, self.s02, self.s03, self.s04, self.s05, self.s06],
                       's (J/kg-K)':[self.s1, self.s2, self.s3, self.s4, self.s5, self.s6],
                       'V (m/s)':[self.V1, self.V2, self.V3, self.V4, self.V5, self.V6]})
        
        self.efficiencies = pd.DataFrame(data={'Diffuser Eff':self.eta_d, 'Compressor Eff':self.eta_c, 'Turbine Eff':self.eta_t, 'Nozzle Eff':self.eta_n},
                                         index=[1])
    
    def print_all_states(self):
        print('------------- Turbojet Engine ------------\n')
        print(self.states)
        print('   \n')
        print('---------- Isentropic Efficiency (eta) -------------\n')
        print(self.efficiencies)
        print('   \n')
        print('---------- Engine Performance --------------\n')
        print(f' Compressor pressure ratio      r_p = {self.rp} \n')
        print(f' Maximum temperature            T_4 = {self.T4} K \n')
        print(f' Specific thrust             F/mdot = {self.F_m} N/(kg/s) \n')
        print(f' TSFC                          TSFC = {self.TSFC} (mg/s/N)\n')
        print(f' Thermal efficiency          eta_th = {self.eta_th} percent \n')
        print(f' Propusive efficiency        eta_PP = {self.eta_PP} percent \n')
        print(f' Exit velocity                   V6 = {self.V6} m/s \n')
        print(f' Cruise Mach number             Ma1 = {self.Ma1}  \n')
        print(f' Axial flow Mach number         Max = {self.Max}  \n')
        print(f' Exit Mach number               Ma6 = {self.Ma6}  \n')
        print(f'     \n')


def run_all():
    turbine = turbojet_engine()
    # turbine.p1 = 50
    turbine.cycle_analysis()
    turbine.perform_analysis()
    turbine.gather_states()
    turbine.print_all_states()
    
if __name__ == "__main__":
    run_all()