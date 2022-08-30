function [R_ave,P_ave]=perfMetric(PC,Neff)
%% Parameters
gmax=10;
numSec=1e3;

%% Initializations
dg=gmax/numSec;
G=dg:dg:gmax;

%% Performance Metrics
P=PC(G);
R=log2(1+P.*G./Neff);

pg=exp(-G);
R_ave=sum(R.*pg)*dg;
P_ave=sum(P.*pg)*dg;

end
