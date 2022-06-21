function P=pwrCtrl(g,L,Pmax,Neff)
%% pwrCtrl
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
P=max(min(L-Neff./g,Pmax),0);

end
