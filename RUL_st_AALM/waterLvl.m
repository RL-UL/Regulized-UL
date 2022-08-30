function L=waterLvl(Pmax,Pmean,Neff)
%% Parameters
eps=1e-15;
Kmax=100;
StrErr='The iteration exceeds %d steps.';

%% Initializations
fun=@(x) ((x*exp(-Neff/x)-Neff*expint(Neff/x))/Pmean-1);% 包络服从瑞利分布，这个初始化是什么意思？expint表示 积分 从x到无穷e-t/t dt       这边应该是把1/epsi看成了整体了x

%% Water Filling
% Find the lower and upper limits
L=Pmax; f=fun(L);
if abs(f)<eps, return;
elseif f>0, Ll=0; Lr=Pmax;
else, Ll=Pmax; k=0; % 这边是找下届
    fun=@(x) (fun(x)-fun(x-Pmax)-1); % 这里的f应该表示功率分配函数的期望等于平均功率这个约束，选择可这个水位线，那么功率分配也就固定了
    while 1, k=k+1; if k>Kmax, error(StrErr,Kmax); end
        Lr=2*Ll; f=fun(Lr);
        if abs(f)<eps, L=Lr; return;
        elseif f>0, break; % 这边是找上界
        end, Ll=Lr;
    end
end

% Bisection Searching
k=0;
while 1, k=k+1; if k>Kmax, error(StrErr,Kmax); end
    Lm=(Ll+Lr)/2; f=fun(Lm); % 这里的f应该表示功率分配函数的期望等于平均功率这个约束，选择可这个水位线，那么功率分配也就固定了
    if abs(f)<eps, L=Lm; return;
    elseif f>0, Lr=Lm;
    else, Ll=Lm;
    end
end
end
