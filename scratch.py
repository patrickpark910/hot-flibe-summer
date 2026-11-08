rho=1940; cp=2400; dT=75; u=2
q_k=241+129; p_k=630; p_v=1260
q_v=q_k*p_v/p_k
flow=q_v*1e6/(rho*cp*dT)
A=flow/u
main=241*2; div=129*2
print(q_v,flow,A,main*1e6/(rho*cp*dT),div*1e6/(rho*cp*dT))
print("1 MW per 0.1 m2 -> min velocity", 100e6/(rho*cp*.1*dT))
print("circum",2*3.14159265*4.62, "10deg", .045*4.62*3.14159265/18)
