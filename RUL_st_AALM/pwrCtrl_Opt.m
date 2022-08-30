function P=pwrCtrl_Opt(g,L,Pmax,Neff)
%% Power Control
P=max(L-Neff./g,0);

end
