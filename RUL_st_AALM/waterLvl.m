function L=waterLvl(Pmax,Pmean,Neff)
%% Parameters
eps=1e-15;
Kmax=100;
StrErr='The iteration exceeds %d steps.';

%% Initializations
fun=@(x) ((x*exp(-Neff/x)-Neff*expint(Neff/x))/Pmean-1);% ������������ֲ��������ʼ����ʲô��˼��expint��ʾ ���� ��x������e-t/t dt       ���Ӧ���ǰ�1/epsi������������x

%% Water Filling
% Find the lower and upper limits
L=Pmax; f=fun(L);
if abs(f)<eps, return;
elseif f>0, Ll=0; Lr=Pmax;
else, Ll=Pmax; k=0; % ��������½�
    fun=@(x) (fun(x)-fun(x-Pmax)-1); % �����fӦ�ñ�ʾ���ʷ��亯������������ƽ���������Լ����ѡ������ˮλ�ߣ���ô���ʷ���Ҳ�͹̶���
    while 1, k=k+1; if k>Kmax, error(StrErr,Kmax); end
        Lr=2*Ll; f=fun(Lr);
        if abs(f)<eps, L=Lr; return;
        elseif f>0, break; % ��������Ͻ�
        end, Ll=Lr;
    end
end

% Bisection Searching
k=0;
while 1, k=k+1; if k>Kmax, error(StrErr,Kmax); end
    Lm=(Ll+Lr)/2; f=fun(Lm); % �����fӦ�ñ�ʾ���ʷ��亯������������ƽ���������Լ����ѡ������ˮλ�ߣ���ô���ʷ���Ҳ�͹̶���
    if abs(f)<eps, L=Lm; return;
    elseif f>0, Lr=Lm;
    else, Ll=Lm;
    end
end
end
