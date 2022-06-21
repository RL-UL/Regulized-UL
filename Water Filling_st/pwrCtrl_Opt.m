function P=pwrCtrl_Opt(g,L,Pmax,Neff)
%% pwrCtrl_Opt
% _ChanGingSuny_ 2019-07-08 v1.0
% 
% Control the transmit power according to the channel gain.
% 
% *Input*
%   |g|			Small scale channel gain
%   |L|         Water level
%   |Pmax|      Maximum transmit power
%   |Neff|      Power of the effective noise at the transmitter
% *Output*
%   |P|     	Transmit power

%% Power Control
P=max(L-Neff./g,0);

end
