function [R_ave,P_ave]=perfMetric(PC,Neff)
%% perfMetric
% _ChanGingSuny_ 2019-07-12 v1.0
% 
% The ergodic channel capacity and transmit power.
% 
% *Input*
%   |PC|        Power control function
%   |Neff|      Power of the effective noise at the transmitter
% *Output*
%   |R_ave|     Ergodic channel capacity
%   |P_ave|     Ergodic transmit power

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
