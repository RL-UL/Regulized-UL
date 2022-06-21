%% Water Filling
% Many thanks for ：C. Sun, D. Liu and C. Yang, “Model-free unsupervised learning for optimization problems with constraints,” in 2019 25th Asia-Pacific Conference on Communications (APCC), pp. 392–397, 2019.
% 
% Find the optimal solution of a water filling problem

clc,clear;
%% Parameters
W=20e6;                     % Bandwidth (Hz)
N0=10^((-173-30)/10);       % Noise power spectrum density (W/Hz)
Pmax=100;                    % Maximum transmit power (W)
Pmean=2;                   % Average transmit power (W)

dist_min=50;                % Minimum distance from user to the BS (m)
dist_max=500;               % Maximum distance from user to the BS (m)
PathLoss=@(x) (10.^-(3.53+3.76*log10(x)));  % Path loss function

%% System Setup
alpha=PathLoss(dist_max);
Neff=W*N0/alpha;

% G=1e-2:1e-2:10;
G=10.^(-2:1e-2:1);

%% Optimal Solution
wl=waterLvl(Pmax,Pmean,Neff); % 这里的w1是1/epsi
PC_Opt=@(g) (pwrCtrl_Opt(g,wl,Pmax,Neff));
[R_ave_Opt,P_ave_Opt]=perfMetric(PC_Opt,Neff);
Power_Opt=PC_Opt(G);
lambda_Opt=max(1./(Neff./G+Pmax)-1/wl,0);% 这里是lambda2
R_Opt=log2(1+G.*Power_Opt/Neff);

%% Learnt Solution
[PNN,LNN,xi,PNN_am,LNN_am,xi_am,R_Lrn_iter,R_Opt_iter,R_Lrn_iter_am,R_equal,T,P_mean,P_max,P_mean_am,P_max_am]=pwrCtrl_MB(Pmean,Pmax,Neff);
% ori
PC_Lrn=@(g) (ForwardProp(PNN,reglr(g)));
[R_ave_Lrn,P_ave_Lrn]=perfMetric(PC_Lrn,Neff);
Power_Lrn=PC_Lrn(G);
lambda_Lrn=ForwardProp(LNN,reglr(G));
R_Lrn=log2(1+G.*Power_Lrn/Neff); % 补充
%am
PC_Lrn_am=@(g) (ForwardProp(PNN_am,reglr(g)));
[R_ave_Lrn_am,P_ave_Lrn_am]=perfMetric(PC_Lrn_am,Neff);
Power_Lrn_am=PC_Lrn_am(G);
lambda_Lrn_am=ForwardProp(LNN_am,reglr(G));
R_Lrn_am=log2(1+G.*Power_Lrn_am/Neff); % 补充
%% Figure Properties
% Colors
%     	  R      G      B
CLR = [ 0.00 , 0.45 , 0.74 ;... 1 Blue
        0.85 , 0.33 , 0.10 ;... 2 Orange-Red
        0.93 , 0.68 , 0.13 ;... 3 Yellow
        0.49 , 0.18 , 0.56 ;... 4 Purple
        0.47 , 0.67 , 0.19 ;... 5 Green
        0.30 , 0.75 , 0.93 ;... 6 Azure
        0.64 , 0.08 , 0.18 ]; % 7 Claret

fBright=@(c,b) (1-b*(1-c));

% Line style
LN	= { '-' , '--' , ':' , '-.' };

% Dot style
MK	= { 'o' , 'square' , '^' , 'v' , '*' , '+' , 'x' };

% Properties
ln=1.5;
mk=2;
ft=14;

%% Plot: Power Control
figure; hold on; grid on; box on;
yyaxis left
hopt=plot(G,Power_Opt,'LineWidth',ln,'Color',CLR(1,:),'DisplayName','Optimal $P(g)$');
hlrn=plot(G,Power_Lrn,'LineWidth',ln,'Color',CLR(6,:),'DisplayName','Learnt $P(g)$');
hlrn_am=plot(G,Power_Lrn_am,'LineWidth',ln,'Color',CLR(3,:),'DisplayName','Learntam $P(g)$');
ylabel('$P(g)$ (W)','Interpreter','latex','FontSize',ft);

yyaxis right
hlamopt=plot(G,lambda_Opt,'LineWidth',ln,'Color',CLR(2,:),'DisplayName','Optimal $\lambda(g)$');
hlamlrn=plot(G,lambda_Lrn,'LineWidth',ln,'Color',CLR(7,:),'DisplayName','Learnt $\lambda(g)$');
hlamlrn_am=plot(G,lambda_Lrn_am,'LineWidth',ln,'Color',CLR(5,:),'DisplayName','Learntam $\lambda(g)$');
ylabel('$\lambda(g)$','Interpreter','latex','FontSize',ft);

h=[hopt,hlrn,hlrn_am,hlamopt,hlamlrn,hlamlrn_am];
legend(h,{},'Interpreter','latex','FontSize',ft,'Box','off','Location','nw');
xlabel('$g$','Interpreter','latex','FontSize',ft);
set(gca,'XScale','log','FontSize',ft);

%% Plot: Channel Capacity
figure; hold on; grid on; box on;
hopt=plot(G,R_Opt,'LineWidth',ln,'Color',CLR(1,:),'DisplayName','Optimal $R(P(g),g)$');
hlrn=plot(G,R_Lrn,'LineWidth',ln,'Color',CLR(6,:),'DisplayName','Learnt $R(P(g),g)$');
hlrn_am=plot(G,R_Lrn_am,'LineWidth',ln,'Color',CLR(3,:),'DisplayName','Learntam $R(P(g),g)$');

h=[hopt,hlrn,hlrn_am];
legend(h,{},'Interpreter','latex','FontSize',ft,'Box','off','Location','best');
xlabel('$g$','Interpreter','latex','FontSize',ft);
ylabel('$R(P(g),g)$ (bits/block)','Interpreter','latex','FontSize',ft);
set(gca,'XScale','log','FontSize',ft);

%% Plot: power mean
figure; hold on; grid on; box on;
hopt=plot(T,P_mean,'LineWidth',ln,'Color',CLR(1,:),'DisplayName','Learnt $R(P(g),g)$');
hlrn=plot(T,P_mean_am,'LineWidth',ln,'Color',CLR(6,:),'DisplayName','Learntam $R(P(g),g)$');

h=[hopt,hlrn];
legend(h,{},'Interpreter','latex','FontSize',ft,'Box','off','Location','best');
xlabel('$T$','Interpreter','latex','FontSize',ft);
ylabel('$pmean$','Interpreter','latex','FontSize',ft);
set(gca,'FontSize',ft);

%% Plot: power max
figure; hold on; grid on; box on;
hopt=plot(T,P_max,'LineWidth',ln,'Color',CLR(1,:),'DisplayName','Learnt $R(P(g),g)$');
hlrn=plot(T,P_max_am,'LineWidth',ln,'Color',CLR(6,:),'DisplayName','Learntam $R(P(g),g)$');

h=[hopt,hlrn];
legend(h,{},'Interpreter','latex','FontSize',ft,'Box','off','Location','best');
xlabel('$T$','Interpreter','latex','FontSize',ft);
ylabel('$pmax$','Interpreter','latex','FontSize',ft);
set(gca,'FontSize',ft);


%% Plot: Channel Capacity via iteration
figure; hold on; grid on; box on;
hopt=plot(T,R_Opt_iter,'LineWidth',ln,'Color',CLR(1,:),'DisplayName','Optimal $R(P(g),g)$');
hlrn=plot(T,R_Lrn_iter,'LineWidth',ln,'Color',CLR(6,:),'DisplayName','Learnt $R(P(g),g)$');
hlrn_am=plot(T,R_Lrn_iter_am,'LineWidth',ln,'Color',CLR(3,:),'DisplayName','Learntam $R(P(g),g)$');
hlrn_equal=plot(T,R_equal,'LineWidth',ln,'Color',CLR(2,:),'DisplayName','equal $R(P(g),g)$');

h=[hopt,hlrn,hlrn_am,hlrn_equal];
legend(h,{},'Interpreter','latex','FontSize',ft,'Box','off','Location','best');
xlabel('$T$','Interpreter','latex','FontSize',ft);
ylabel('$R(P(g),g)$ (bits/block)','Interpreter','latex','FontSize',ft);
set(gca,'FontSize',ft);









