function P=pwrCtrl(g,L,Pmax,Neff)
%% Power Control
P=max(min(L-Neff./g,Pmax),0);

end
