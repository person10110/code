clc;
clf; 
% p=[];
% q=[]; 
% B1 = [];
% B2 = [];
% B3 = [];
% B4 = [];
% B5 = [];
% B6 = [];
% B7 = [];
% B8 = [];
% B9 = [];
% B10 = [];
% B11 = [];
% B12 = [];
% B13 = [];
% B14 = [];
% B15 = [];
% B16 = [];
% 
% PP1 = [];
% PP2 = [];
% PP3 = [];
% PP4 = [];
% PP5 = [];
% PP6 = [];
% PP7 = [];
% PP8 = [];
% PP9 = [];
% PP10 = [];
% PP11 = [];
% PP12 = [];
% PP13 = [];
% PP14 = [];
% PP15 = [];
% PP16 = [];
% q=[];
% syms x
% 
% for o=2:13
% 
% 
% Fit5 = fit(table2array(WeatherDataS3(:,1)),(table2array(WeatherDataS3(:,o))),'sin5');
% Fit6 = fit(table2array(WeatherDataS3(:,1)),(table2array(WeatherDataS3(:,o))),'sin6');
% Fit7 = fit(table2array(WeatherDataS3(:,1)),(table2array(WeatherDataS3(:,o))),'sin7');
% Fit8 = fit(table2array(WeatherDataS3(:,1)),(table2array(WeatherDataS3(:,o))),'sin8');
% 
% Fit13 = fit((table2array(WeatherDataS3(:,o))),(table2array(WeatherDataS3(:,o+14))),'sin5');
% Fit14 = fit((table2array(WeatherDataS3(:,o))),(table2array(WeatherDataS3(:,o+14))),'sin6');
% Fit15 = fit((table2array(WeatherDataS3(:,o))),(table2array(WeatherDataS3(:,o+14))),'sin7');
% Fit16 = fit((table2array(WeatherDataS3(:,o))),(table2array(WeatherDataS3(:,o+14))),'sin8');
% 
% 
% 
% CL = 0.7; 
% z = icdf('Normal',1-(1-CL)/2,0,1); 
% 
% M5 = confint(Fit5, CL); 
% N5 = M5(1,:).*M5(2,:);
% i5 = find(N5<0); 
% p5 = [length(i5),length(N5)];
% se5 = (M5(2,:)-M5(1,:))/z;
% b5 = (M5(2,:)+M5(1,:))./2; 
% P5 = 1 - cdf('T', b5./se5, length(table2array(WeatherDataS3(:,1)))-length(N5));
% P_5 = P5 - 0.3*ones(length(P5),1)';
% PP5 = [PP5; length(find(P_5>0))/(length(P5))];
% B5 = [B5; b5]; 
% 
% M6 = confint(Fit6, CL); 
% N6 = M6(1,:).*M6(2,:);
% i6 = find(N6<0); 
% p6 = [length(i6),length(N6)];
% se6 = (M6(2,:)-M6(1,:))/z;
% b6 = (M6(2,:)+M6(1,:))./2; 
% P6 = 1 - cdf('T', b6./se6, length(table2array(WeatherDataS3(:,1)))-length(N6));
% P_6 = P6 - 0.3*ones(length(P6),1)';
% PP6 = [PP6; length(find(P_6>0))/(length(P6))];
% B6 = [B6; b6]; 
% 
% M7 = confint(Fit7, CL); 
% N7 = M7(1,:).*M7(2,:);
% i7 = find(N7<0); 
% p7 = [length(i7),length(N7)];
% se7 = (M7(2,:)-M7(1,:))/z;
% b7 = (M7(2,:)+M7(1,:))./2; 
% P7 = 1 - cdf('T', b7./se7, length(table2array(WeatherDataS3(:,1)))-length(N7));
% P_7 = P7 - 0.3*ones(length(P7),1)';
% PP7 = [PP7; length(find(P_7>0))/(length(P7))];
% B7 = [B7; b7]; 
% 
% M8 = confint(Fit8, CL); 
% N8 = M8(1,:).*M8(2,:);
% i8 = find(N8<0); 
% p8 = [length(i8),length(N8)];
% se8 = (M8(2,:)-M8(1,:))/z;
% b8 = (M8(2,:)+M8(1,:))./2; 
% P8 = 1 - cdf('T', b8./se8, length(table2array(WeatherDataS3(:,1)))-length(N8));
% P_8 = P8 - 0.3*ones(length(P8),1)';
% PP8 = [PP8; length(find(P_8>0))/(length(P8))];
% B8 = [B8; b8];
% 
% 
% M13 = confint(Fit13, CL); 
% N13 = M13(1,:).*M13(2,:);
% i13 = find(N13<0); 
% p13 = [length(i13),length(N13)];
% se13 = (M13(2,:)-M13(1,:))/z;
% b13 = (M13(2,:)+M13(1,:))./2; 
% P13 = 1 - cdf('T', b13./se13, length(table2array(WeatherDataS3(:,1)))-length(N13));
% P_13 = P13 - 0.3*ones(length(P13),1)';
% PP13 = [PP13; length(find(P_13>0))/(length(P13))];
% B13 = [B13; b13]; 
% 
% M14 = confint(Fit14, CL); 
% N14 = M14(1,:).*M14(2,:);
% i14 = find(N14<0); 
% p14 = [length(i14),length(N14)];
% se14 = (M14(2,:)-M14(1,:))/z;
% b14 = (M14(2,:)+M14(1,:))./2; 
% P14 = 1 - cdf('T', b14./se14, length(table2array(WeatherDataS3(:,1)))-length(N14));
% P_14 = P14 - 0.3*ones(length(P14),1)';
% PP14 = [PP14; length(find(P_14>0))/(length(P14))];
% B14 = [B14; b14]; 
% 
% M15 = confint(Fit15, CL); 
% N15 = M15(1,:).*M15(2,:);
% i15 = find(N15<0); 
% p15 = [length(i15),length(N15)];
% se15 = (M15(2,:)-M15(1,:))/z;
% b15 = (M15(2,:)+M15(1,:))./2; 
% P15 = 1 - cdf('T', b15./se15, length(table2array(WeatherDataS3(:,1)))-length(N15));
% P_15 = P15 - 0.3*ones(length(P15),1)';
% PP15 = [PP15; length(find(P_15>0))/(length(P15))];
% B15 = [B15; b15]; 
% 
% M16 = confint(Fit16, CL); 
% N16 = M16(1,:).*M16(2,:);
% i16 = find(N16<0); 
% p16 = [length(i16),length(N16)];
% se16 = (M16(2,:)-M16(1,:))/z;
% b16 = (M16(2,:)+M16(1,:))./2; 
% P16 = 1 - cdf('T', b16./se16, length(table2array(WeatherDataS3(:,1)))-length(N16));
% P_16 = P16 - 0.3*ones(length(P16),1)';
% PP16 = [PP16; length(find(P_16>0))/(length(P16))];
% B16 = [B16; b16]; 
% 
% p = [p; p5 p6 p7 p8];
% 
% 
% q = [q; p13 p14 p15 p16];
% 
% end
% 
% 
% s = [PP1 PP2 PP3 PP4 PP5 PP6 PP7 PP8];
% s1 = [PP9 PP10 PP11 PP12 PP13 PP14 PP15 PP16]; 
% [m1,j1] = min(s(1,:));
% [m2,j2] = min(s(2,:));
% [m3,j3] = min(s(3,:));
% [m4,j4] = min(s(4,:));
% [m5,j5] = min(s(5,:));
% [m6,j6] = min(s(6,:));
% [m7,j7] = min(s(7,:));
% [m8,j8] = min(s(8,:));
% [m9,j9] = min(s(9,:));
% [m10,j10] = min(s(10,:));
% [m11,j11] = min(s(11,:));
% [m12,j12] = min(s(12,:));
% 
% [l1,h1] = min(s1(1,:));
% [l2,h2] = min(s1(2,:));
% [l3,h3] = min(s1(3,:));
% [l4,h4] = min(s1(4,:));
% [l5,h5] = min(s1(5,:));
% [l6,h6] = min(s1(6,:));
% [l7,h7] = min(s1(7,:));
% [l8,h8] = min(s1(8,:));
% [l9,h9] = min(s1(9,:));
% [l10,h10] = min(s1(10,:));
% [l11,h11] = min(s1(11,:));
% [l12,h12] = min(s1(12,:));
% 
% j = [j1, j2, j3, j4, j5, j6, j7, j8, j9, j10, j11, j12];
% h = [h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12];
% m = [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12];
% l = [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12];
% 
% J1 = j(:,4);
% J2 = j(:,5);
% J3 = j(:,6);
% J4 = j(:,7);
% J5 = j(:,8);
% J6 = j(:,9);
% 
% H1 = h(:,4);
% H2 = h(:,5);
% H3 = h(:,6);
% H4 = h(:,7);
% H5 = h(:,8);
% H6 = h(:,9);
% 
% 
% M1 = m(:,4);
% M2 = m(:,5);
% M3 = m(:,6);
% M4 = m(:,7);
% M5 = m(:,8);
% M6 = m(:,9);
% 
% 
% L1 = l(:,4);
% L2 = l(:,5);
% L3 = l(:,6);
% L4 = l(:,7);
% L5 = l(:,8);
% L6 = l(:,9);
% 
% J = [J1, J2, J3, J4, J5, J6]'; 
% H = [H1, H2, H3, H4, H5, H6]';
% L = [L1, L2, L3, L4, L5, L6]';
% M = [M1, M2, M3, M4, M5, M6]';
% 
% 
% P1 = p(4, J1*2); 
% P11 = p(4, J1*2 -1 ); 
% 
% P2 = p(5, J2*2); 
% P21 = p(5, J2*2 -1 ); 
% 
% P3 = p(6, J3*2); 
% P31 = p(6, J3*2 -1 ); 
% 
% P4 = p(7, J4*2); 
% P41 = p(7, J4*2 -1 ); 
% 
% P5 = p(8, J5*2); 
% P51 = p(8, J5*2 -1 ); 
% 
% P6 = p(9, J6*2); 
% P61 = p(9, J6*2 -1 ); 
% 
% Q1 = q(4, H1*2); 
% Q11 = q(4, H1*2 -1 ); 
% 
% Q2 = q(5, H2*2); 
% Q21 = q(5, H2*2 -1 ); 
% 
% Q3 = q(6, H3*2); 
% Q31 = q(6, H3*2 -1 ); 
% 
% Q4 = q(7, H4*2); 
% Q41 = q(7, H4*2 -1 ); 
% 
% Q5 = q(8, H5*2); 
% Q51 = q(8, H5*2 -1 ); 
% 
% Q6 = q(9, H6*2); 
% Q61 = q(9, H6*2 -1 ); 
% 
% pn = [P1, P2, P3, P4, P5, P6]';
% qn = [Q1, Q2, Q3, Q4, Q5, Q6]';
% 
% mn1 = [P11, P21, P31, P41, P51, P61]';
% mn2 = [Q11, Q21, Q31, Q41, Q51, Q61]';
% 
% Mo = [4, 5, 6, 7, 8, 9]'; 
% 
% R_t = [Mo J M pn mn1]; 
% R_r = [Mo H L qn mn2]; 
% 
% R_t
% 
% R_r
syms x
FKE1 = fit(table2array(WeatherDataS3(:,1)),(table2array(WeatherDataS3(:,Mo(1,:)+1))),'sin6');
FKE2 = fit(table2array(WeatherDataS3(:,1)),(table2array(WeatherDataS3(:,Mo(2,:)+1))),'sin8');
FKE3 = fit(table2array(WeatherDataS3(:,1)),(table2array(WeatherDataS3(:,Mo(3,:)+1))),'sin8');
FKE4 = fit(table2array(WeatherDataS3(:,1)),(table2array(WeatherDataS3(:,Mo(3,:)+1))),'sin8');
FKE5 = fit(table2array(WeatherDataS3(:,1)),(table2array(WeatherDataS3(:,Mo(4,:)+1))),'sin8');
FKE6 = fit(table2array(WeatherDataS3(:,1)),(table2array(WeatherDataS3(:,Mo(4,:)+1))),'sin8');

GKE1 = fit((table2array(WeatherDataS3(:,Mo(1,:)+1))),(table2array(WeatherDataS3(:,Mo(1,:)+15))),'sin5');
GKE2 = fit((table2array(WeatherDataS3(:,Mo(2,:)+1))),(table2array(WeatherDataS3(:,Mo(2,:)+15))),'sin7');
GKE3 = fit(table2array(WeatherDataS3(:,Mo(3,:)+1)),(table2array(WeatherDataS3(:,Mo(3,:)+15))),'sin6'); 
GKE4 = fit((table2array(WeatherDataS3(:,Mo(4,:)+1))),(table2array(WeatherDataS3(:,Mo(4,:)+15))),'sin6');
GKE5 = fit((table2array(WeatherDataS3(:,Mo(5,:)+1))),(table2array(WeatherDataS3(:,Mo(5,:)+15))),'sin5');
GKE6 = fit((table2array(WeatherDataS3(:,Mo(6,:)+1))),(table2array(WeatherDataS3(:,Mo(6,:)+15))),'sin7');

F_1 = FKE1(x);
F_2 = FKE2(x);
F_3 = FKE3(x);
F_4 = FKE4(x);
F_5 = FKE5(x);
F_6 = FKE6(x);

G_1 = GKE1(FKE1(x));
G_2 = GKE2(FKE2(x));
G_3 = GKE3(FKE3(x));
G_4 = GKE4(FKE4(x));
G_5 = GKE5(FKE5(x));
G_6 = GKE6(FKE6(x));

Y = []; 
X = []; 

for x = 2023:2072
a_1 = [linspace(0,1,200), linspace(1,2,200), linspace(2,3,100), linspace(-1,0,100)] ;
b_1 = [linspace(-15,-10,100), linspace(-10,-5,200), linspace(-5,0,300), linspace(0,5,100), linspace(5,10,50), linspace(10,15,50)];

a1 = randsample(a_1, 1); 
b1 = randsample(b_1, 1); 
t1=5.49565217391304+a1;
t2=8.12971014492754+a1;
t3=10.7507246376812+a1;
t4=12.3405797101449+a1;
t5=12.2065217391304+a1;
t6=10.3673913043478+a1;

r1=91.6064516129032+b1;
r2=81.9161290322581+b1;
r3=87.5709677419355+b1;
r4=103.739784946237+b1;
r5=120.070430107527+b1;
r6=130.138172043011+b1;

y11 = 0.93.^(FKE1(x)*t1)*(1+(GKE1(FKE1(x))))*r1 + 0.93.^(FKE2(x)*t2)*(1+(GKE2(FKE2(x))))*r2 + 0.93.^(FKE3(x)*t3)*(1+GKE3(FKE3(x)))*r3 + 0.93.^(FKE4(x)*t4)*(1+GKE4(FKE4(x)))*r4 + 0.93.^(FKE5(x)*t5)*(1+GKE5(FKE5(x)))*r5 + 0.93.^(FKE6(x)*t6)*(1+GKE6(FKE6(x)))*r6;
Y = [Y; min(y11/500, 1.25)];
X = [X; a1 b1]; 
end
X;
Y;
X1 = mean(X(:,1));
X2 = mean(X(:,2));
syms t

m1 = 0.2; %Initial Growth Rate
m2 = 0.00; %Final Growth Rate
r_r = 0.00001; %Rate at which growth rate decreases 
r_t = 2*(m1 - m2)./(1+exp(r_r*t)) + m2; 

wm = 0.482; 
D_m = 0.6; 
a = (D_m-0.005)/0.005; 
D_t = D_m./(1+a*exp(-r_t.*t)); 


m3 = 0.2; 
m4 = 0.00; 
q_r = 0.00001; 
q_t = 2*(m3 - m4)./(1+exp(q_r*t)) + m4 ; 

H_m = 20; 
b = (H_m-0.2)/0.2; 
H_t = H_m./(1+b*exp(-q_t.*t));

C_1 = 1.2*(1-wm)*3.6663*(3.28*39.37^2)/2.205; 

W_CO2 = 0.15 * C_1 * H_t.*(D_t.^2); 
 
dq_t = -(2*(m3-m4)*(q_r.*exp(q_r.*t)))/((1+exp(q_r.*t)).^2); 
dr_t = -(2*(m1-m2)*(r_r.*exp(r_r.*t)))/((1+exp(r_r.*t)).^2);

n = b*(dq_t.*t + q_t)...
    .*exp(-q_t.*t).*((1+a.*exp(-r_t.*t)).^2) ... 
+ 2.*(dr_t.*t + r_t).*a.*exp(-r_t.*t).*(1+b*exp(-q_t.*t)).*(1+a.*exp(-r_t.*t));

d = (1+b*exp(-q_t.*t)).*(1+a.*exp(-r_t.*t)).^2; 

dW_CO2 = W_CO2*(n/d);

f_r = 5*1.05.^t - 4*1.04.^t;

t_r = D_t.^2*H_t.*3;  

dW_CO21 = matlabFunction(dW_CO2);
WCO21 = matlabFunction(W_CO2); 
Ht = matlabFunction(H_t); 
Dt = matlabFunction(D_t);
f_rt = matlabFunction(f_r);
t_rt = matlabFunction(t_r);


%Base Case 

s=0.01;                                                                                      
m = [];
y = 0;
   for t=s:s:11                             
    k1 = dW_CO21(t);
    k2 = dW_CO21(t)+0.5*s*k1;
    k3 = dW_CO21(t)+0.5*s*k2;
    k4 = dW_CO21(t)+k3*s;
    y = y + (1/6)*(k1+2*k2+2*k3+k4)*s;
    y1 = (1/6)*(k1+2*k2+2*k3+k4)*s;
    m = [m; t y y1 y/t]; 
   end

%PPPP(m(:,1), m(:,2))


B = m((1/s):(1/s):end,:);

[o, j] = max(B); 

results = [j(3) o(2) j(4) o(4)];  

%results
% 
% 
%With probability differences (water availability)  

%Impacts on growth: 0.6 (20%, 10 times), 0.7 (30%, 15 times), 0.8 (30%, 15 times), 0.9 (10%, 5 times), 1 (10%, 5 times) 

t1=50; 
s1 = 0.01;                                                                                      
n1 = [];      
   for t=s1:s1:t1                             
    k1 = dW_CO21(t);
    k2 = dW_CO21(t)+0.5*s1*k1;
    k3 = dW_CO21(t)+0.5*s1*k2;
    k4 = dW_CO21(t)+k3*s1;
    y1 = (1/6)*(k1+2*k2+2*k3+k4)*s1;
    n1 = [n1;t y1]; 
   end
   
n1(:,2); 
Y; 
u = repelem(Y',100);
u;
length(n1(:,2));
length(Y);

PP = []; 
ww2 = 1;
for n=1:50
vec = [0.95, 1, 1, 1, 1, 1, 1, 1, 1, 1]; 
ww1 = randsample(vec,1); 
ww2 = ww2*ww1; 
PP = [PP; ww1 ww2]; 
end

PPP = [];
www2 = 1; 
for n=1:50
    
vvec = [0.95, 1, 1, 1, 1]; 
www1 = randsample(vvec,1); 
www2 = www2*www1; 
PPP = [PPP; www1 www2]; 
end

PPPP = []; 
wwww2 = 1; 
for n=1:50
vvvec = [0.95, 0.95, 1, 1, 1]; 
wwww1 = randsample(vvvec,1); 
wwww2 = wwww2*wwww1; 
PPPP = [PPPP; wwww1 wwww2]; 
end


u1 = repelem(PP(:,2)',100);
u1;

u2 = repelem(PPP(:,2)',100);
u2;

u3 = repelem(PPPP(:,2)',100);
u3;

n3 = (u'.*u1').*n1(:,2);

nx3 = (u'.*u2').*n1(:,2);

nxx3 = (u'.*u3').*n1(:,2);




nn3 = u'.*n1(:,2); 


% nd1 = []; 
% nd2 = []; 



n4 = []; 
for m = 1:50 
n5 = sum(n3(m*100-99:m*100)); 
n4 = [n4;m/100 n5]; 
end

nn4 = []; 
for m = 100:100:5000 
nn5 = sum(nn3(m-99:m)); 
nn4 = [nn4;m/100 nn5]; 
end


nu2 = []; 
for m = 100:100:5000 
nu5 = sum(nx3(m-99:m)); 
nu2 = [nu2;m/100 nu5]; 
end

nu3 = []; 

for m = 100:100:5000 
nu6 = sum(nxx3(m-99:m)); 
nu3 = [nn4;m/100 nu6]; 
end


[o1, j1] = max(n4);
[oo1, jj1] = max(nn4); 
[ooo1, jjj1] = max(nu2); 
[oooo1, jjjj1] = max(nu3); 

n9=[]; 
n7=0; 
for n=1:50   
    n7 = n7+n4(n,:);
    
    n9 = [n9; n n7 n7/n]; 
end 


nn9=[]; 
nn7=0; 
for n=1:50   
    nn7 = nn7+nn4(n,:);
    
    nn9 = [nn9; n nn7 nn7/n]; 
end 

nu9=[]; 
nu7=0; 
for n=1:50   
    nu7 = nu7+nu2(n,:);
    
    nu9 = [nu9; n nu7 nu7/n]; 
end 

nu10=[]; 
nu8=0; 
for n=1:50   
    nu8 = nu8+nu3(n,:);
    
    nu10 = [nu10; n nu8 nu8/n]; 
end 


n9
nn9
nu9

% %fff = []; 
% syms e
% %for t=1:50 
%    g55 = solve(0.15 * C_1 *e^3*(Ht(t))*(Dt(t)^2) == n9(t,3), e);
%    %fff = [fff; g]; 
% %end
% 
% %ffg = []; 
% 
% %for t=1:50 
%    g66 = solve(0.15 * C_1 *e^3*(Ht(t))*(Dt(t)^2) == nn9(t,3), e);
%    %ffg = [ffg; g]; 
% %end

% www = []; 
% for t=1:50
% W = Dt(t);
% WW = Ht(t); 
% www = [www; W WW]; 
% end

% [o2,j2] = max(n9); 
% [oo2, jj2] = max(nn9);
% [ooo2 ,jjj2] = max(nu9);
% [oooo2, jjjj2] = max(nu10);
% %plot(n9(:,1), n9(:,3));
% Results1 = [j1(2),  o2(3), j2(5), o2(5)] 
% Results2 = [jj1(2),  oo2(3), jj2(5), oo2(5)]
% Results3 = [jjj1(2),  ooo2(3), jjj2(5), ooo2(5)]
% Results4 = [jjjj1(2),  oooo2(3), jjjj2(5), oooo2(5)]; 


% figure
% hold on
% scatter(table2array(WeatherDataS12(:,1)),table2array(WeatherDataS12(:,Mo(1,:)+1)), 'filled')
% plot(x, FKE1(x))
% title('April Temperature Variability')
% hold off
% 
% figure
% hold on
% scatter(table2array(WeatherDataS12(:,1)),table2array(WeatherDataS12(:,Mo(2,:)+1)), 'filled')
% plot(x, FKE2(x))
% title('May Temperature Variability')
% hold off
% 
% figure
% hold on
% scatter(table2array(WeatherDataS12(:,1)),table2array(WeatherDataS12(:,Mo(3,:)+1)), 'filled')
% plot(x, FKE3(x))
% title('June Temperature Variability')
% hold off
% 
% figure
% hold on
% scatter(table2array(WeatherDataS12(:,1)),table2array(WeatherDataS12(:,Mo(4,:)+1)), 'filled')
% plot(x, FKE4(x))
% title('July Temperature Variability')
% hold off
% 
% figure
% hold on
% scatter(table2array(WeatherDataS12(:,1)),table2array(WeatherDataS12(:,Mo(5,:)+1)), 'filled')
% plot(x, FKE5(x))
% title('August Temperature Variability')
% hold off
% 
% figure
% hold on
% scatter(table2array(WeatherDataS12(:,1)),table2array(WeatherDataS12(:,Mo(6,:)+1)), 'filled')
% plot(x, FKE6(x))
% title('September Temperature Variability')
% hold off
% 
% figure
% hold on
% scatter(table2array(WeatherDataS12(:,1)),table2array(WeatherDataS11(:,Mo(1,:)+1)), 'filled')
% plot(x, GKE1(FKE1(x)))
% title('April Rain Variability')
% hold off
% 
% figure
% hold on
% scatter(table2array(WeatherDataS12(:,1)),table2array(WeatherDataS11(:,Mo(2,:)+1)), 'filled')
% plot(x, GKE2(FKE2(x)))
% title('May Rain Variability')
% hold off
% 
% figure
% hold on
% scatter(table2array(WeatherDataS12(:,1)),table2array(WeatherDataS11(:,Mo(3,:)+1)), 'filled')
% plot(x, GKE3(FKE3(x)))
% title('June Rain Variability')
% hold off
% 
% figure
% hold on
% scatter(table2array(WeatherDataS12(:,1)),table2array(WeatherDataS11(:,Mo(4,:)+1)), 'filled')
% plot(x, GKE4(FKE4(x)))
% title('July Rain Variability')
% hold off
% 
% figure
% hold on
% scatter(table2array(WeatherDataS12(:,1)),table2array(WeatherDataS11(:,Mo(5,:)+1)), 'filled')
% plot(x, GKE5(FKE5(x)))
% title('August Rain Variability')
% hold off
% 
% figure
% hold on
% scatter(table2array(WeatherDataS12(:,1)),table2array(WeatherDataS11(:,Mo(6,:)+1)), 'filled')
% plot(x, GKE6(FKE6(x)))
% title('September Rain Variability')
% hold off









