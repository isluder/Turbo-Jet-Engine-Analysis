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
    
    

#Still needs converted to python is the graphing

# %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# %   Plot the processes on a T-s diagram
#     figure (1)
# %   Draw constant pressure lines crossing
#     Tp = 100:10:2000;
#     %
#     %   Stagnation pressure lines
#     s_p1 = s1 + cp*log(Tp/T01);
#     s_p2 = s2 + cp*log(Tp/T02);
#     s_p3 = s3 + cp*log(Tp/T03);
#     s_p4 = s4 + cp*log(Tp/T04);
#     s_p5 = s5 + cp*log(Tp/T05);
#     s_p6 = s6 + cp*log(Tp/T06);
#     plot(s_p1, Tp, 'k--',s_p6, Tp, 'k--')
#     hold on
# %     plot(s_p3,Tp, 'k--', s_p4, Tp, 'k--')
# %     plot(s_p5, Tp, 'k--',s_p6, Tp, 'k--')
#     axis([0 1200 100 1600])
#     xlabel('Entropy s (J/kg-K)','FontSize', 14)
#     ylabel('Temperature T (K)','FontSize', 14)
#     %title('Non-Ideal Turbojet Engine','FontSize', 14)
#     grid on
#     %
#     %   Static pressure lines
#     s_p1 = s1 + cp*log(Tp/T1);
#     s_p2 = s2 + cp*log(Tp/T2);
#     s_p3 = s3 + cp*log(Tp/T3);
#     s_p4 = s4 + cp*log(Tp/T4);
#     s_p5 = s5 + cp*log(Tp/T5);
#     s_p6 = s6 + cp*log(Tp/T6);
#     plot(s_p1, Tp, 'b--',s_p2, Tp, 'b--')
#     hold on
#     plot(s_p3,Tp, 'b--', s_p4, Tp, 'b--')
#     plot(s_p5, Tp, 'b--',s_p6, Tp, 'b--')
# %
# %   Plot process lines, connecting the static states
# %
#     plot([s1 s2], [T1 T2],'m', 'LineWidth',2.0)
#     plot([s2 s3], [T2 T3],'m', 'LineWidth',2.0)
#     %plot([s3 s4], [T3 T4],'m', 'LineWidth',2.0)
#     plot([s4 s5], [T4 T5],'m', 'LineWidth',2.0)
#     plot([s5 s6], [T5 T6],'m', 'LineWidth',2.0)
# %
# %------------------
# %   Plot the process line in burner from 3 to 4 with P4/p3 = pi_b
#     pbi =linspace(p3,p4,10);
#     Tbi = linspace(T3,T4,10);
#     sb(1) = s3;
#     for i = 2:10
#         sb(i) = sb(i-1) +  cp*log(Tbi(i)/Tbi(i-1))-R*log(pbi(i)/pbi(i-1));
#     end
#     plot(sb, Tbi,'m', 'LineWidth',2.0)
#     %---------------
# %   Connect the static to stgnation states
# %
#     plot([s1 s01], [T1 T01],'r', 'LineWidth',0.5)
#     plot([s2 s02], [T2 T02],'r', 'LineWidth',0.5)
#     plot([s3 s03], [T3 T03],'r', 'LineWidth',0.5)
#     plot([s4 s04], [T4 T04],'r', 'LineWidth',0.5)
#     plot([s5 s05], [T5 T05],'r', 'LineWidth',0.5)
#     plot([s6 s06], [T6 T06],'r', 'LineWidth',0.5)
# %
# %   Plot the stagnation states
# %
#     plot(s01, T01, 'o', s02,T02, 'o', 'MarkerFaceColor','r')
#     plot(s03, T03, 'o', s04,T04, 'o', 'MarkerFaceColor','r')
#     plot(s05, T05, 'o', s06,T06, 'o', 'MarkerFaceColor','r')
# %
# %   Plot the static state states
# %
#     plot(s1, T1, 'o', s2,T2, 'o', 'MarkerFaceColor','b')
#     plot(s3, T3, 'o', s4,T4, 'o', 'MarkerFaceColor','b')
#     plot(s5, T5, 'o', s6,T6, 'o', 'MarkerFaceColor','b')
# %
# %   Label 6 states
#     text(s1-30,T1-50,'1','fontSize', 14)
#     text(s01-60,T01+50,'01','fontSize', 14)
#     text(s2+20,T2,'2','fontSize', 14)
#     text(s3-40,T3+40,'3','fontSize', 14) 
#     text(s4-40,T4+40,'4','fontSize', 14)
#     text(s5-50,T5+30,'5','fontSize', 14) 
#     text(s6,T6-50,'6','fontSize', 14)
#     text(s06+10,T06-50,'06','fontSize', 14)
# %
#     hold off
# %
# %   End of the code
# %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
