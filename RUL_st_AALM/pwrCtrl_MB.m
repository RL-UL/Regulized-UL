function [PNN,LNN,xi,PNN_am,LNN_am,xi_am,R_ave_lrn,R_ave_opt,R_ave_lrn_am,R_ave_equal,T,P_mean,P_max,P_mean_am,P_max_am]=pwrCtrl_MB(Pmean,Pmax,Neff,PNN0,LNN0,xi0)
%% Parameters
% lrnRtxi=0.1;
% lrnRtxi_am=0.1;
% nnLrnRtP=0.01;
% nnLrnRtL=0.0077;

lrnRtxi=1;
lrnRtxi_am=1;
nnLrnRtP=0.01;
nnLrnRtL=0.0077;

ScalarX=1;
ScalarL=1;

Tmax=1e3;
batchSz=1e2;
numEpch=1e3;

%% Initializations
if nargin<4 || isempty(PNN0) || isempty(LNN0) || isempty(xi0) % nargin是“number of input arguments”的缩写，判断输入变量个数，可能是为了判断是不是有预训练的模型，我觉得这里应该用&&
    PNN=PNNGen; LNN=LNNGen; % 初始化P和lambda网络
    PNN_am=PNNGen; LNN_am=LNNGen; % 初始化P和lambda网络
%     xi=0.5+0.5*rand; % 初始化因子xi
    xi=0;
    xi_am=0;
else
    PNN=PNN0; LNN=LNN0; xi=xi0;
    PNN_am=PNN0_am; LNN_am=LNN0_am; xi_am=xi0_am;
end

% if nargin<4 || isempty(PNN0) || isempty(LNN0) || isempty(xi0) % nargin是“number of input arguments”的缩写，判断输入变量个数，可能是为了判断是不是有预训练的模型，我觉得这里应该用&&
%     PNN_am=PNNGen; LNN_am=LNNGen; % 初始化P和lambda网络
% %     xi=0.5+0.5*rand; % 初始化因子xi
%     xi_am=0;
% else
%     PNN_am=PNN0_am; LNN_am=LNN0_am; xi_am=xi0_am;
% end

%% optimal
wl=waterLvl(Pmax,Pmean,Neff); % 这里的w1是1/epsi
PC_Opt=@(g) (pwrCtrl_Opt(g,wl,Pmax,Neff));

%% SGD  
t=0; T=(1:Tmax);R_ave_lrn=zeros(1,Tmax);R_ave_opt=zeros(1,Tmax);R_ave_lrn_am=zeros(1,Tmax);R_ave_equal=zeros(1,Tmax);P_mean = zeros(1,Tmax);P_mean_am = zeros(1,Tmax);P_max = zeros(1,Tmax);P_max_am = zeros(1,Tmax);
while t<Tmax, t=t+1;
    t
    H=randn(2,batchSz); G=sum(H.*H,1)/2; % 随机信道的幅度
    
    for nepoc=1:numEpch
        %ori
        P=ForwardProp(PNN,reglr(G)); % reglr是归一化输入，因为最大取10，这个时候概率已经很低了，按照10归一化就可以了
        L=ForwardProp(LNN,reglr(G));
        [gradP,gradL,gradX,R]=gradLossFunc(Pmean,Pmax,Neff,G,P,L/ScalarL,xi/ScalarX);
        
        BackwardProp(PNN,-gradP,nnLrnRtP);
        BackwardProp(LNN,-gradL,nnLrnRtL);
        
        gradX_ave=sum(gradX,2)/batchSz;
        dxi=lrnRtxi*gradX_ave;
        xi_nxt=xi+dxi;
        xi=xi_nxt;
        xi = max(xi,0);
        
        % am
        P_am=ForwardProp(PNN_am,reglr(G)); % reglr是归一化输入，因为最大取10，这个时候概率已经很低了，按照10归一化就可以了
        L_am=ForwardProp(LNN_am,reglr(G));
        [gradP_am,gradL_am,gradX_am,R_am]=gradLossFunc_am(Pmean,Pmax,Neff,G,P_am,L_am,xi_am,lrnRtxi_am);
        
        BackwardProp(PNN_am,-gradP_am,nnLrnRtP);
        BackwardProp(LNN_am,-gradL_am,nnLrnRtL);
        
        gradX_ave_am=sum(gradX_am,2)/batchSz;
        dxi_am=lrnRtxi_am*gradX_ave_am;
        xi_nxt_am=xi_am+dxi_am;
        xi_am=xi_nxt_am;
        xi_am = max(xi_am,0);
        
    end
    R_ave_lrn(1,t)=sum(R,2)/batchSz;
    R_ave_lrn_am(1,t)=sum(R_am,2)/batchSz;
    
    %optimal
    Power_Opt=PC_Opt(G);
    R_Opt=log2(1+G.*Power_Opt/Neff);
    R_ave_opt(1,t)=sum(R_Opt,2)/batchSz;
    
    P_mean(1,t) = mean(P);
    P_max(1,t) = max(P);
    
    P_mean_am(1,t) = mean(P_am);
    P_max_am(1,t) = max(P_am);
    
    %equal
    R_equal = log2(1+G.*Pmean/Neff);
    R_ave_equal(1,t) = sum(R_equal,2)/batchSz;
    
end
end



function [gradP,gradL,gradX,R]=gradLossFunc(Pmean,Pmax,Neff,g,P,L,xi)
%% gradLossFunc
% _ChanGingSuny_ 2019-07-12 v1.0
% 
% The gradient of the loss function w.r.t. the transmit power and the
% Lagrange multiplier.
% 
% *Input*
%   |Pmean|     Average transmit power (W)
%   |Pmax|     	Maximum transmit power (W)
%   |Neff|      Power of the effective noise at the transmitter
%   |g|         Channel gain
%   |P|         Transmit power (W)
%   |L|         State variant Lagrange multiplier
%   |xi|        Lagrange multiplier
% *Output*
%   |gradP|     Gradient w.r.t. transmit power
%   |gradL|     Gradient w.r.t. the state variant Lagrange multiplier
%   |gradX|     Gradient w.r.t. Lagrange multiplier
%   |R|      	Channel capacity

%% Gradients
gradP=1./(Neff./g+P)-xi;
gradL=(P-Pmax)/Pmax;
gradX=(P-Pmean)/Pmean;
R=log2(1+g.*P/Neff);

end

function [gradP,gradL,gradX,R]=gradLossFunc_am(Pmean,Pmax,Neff,g,P,L,xi,lr)
%% gradLossFunc
% _ChanGingSuny_ 2019-07-12 v1.0
% 
% The gradient of the loss function w.r.t. the transmit power and the
% Lagrange multiplier.
% 
% *Input*
%   |Pmean|     Average transmit power (W)
%   |Pmax|     	Maximum transmit power (W)
%   |Neff|      Power of the effective noise at the transmitter
%   |g|         Channel gain
%   |P|         Transmit power (W)
%   |L|         State variant Lagrange multiplier
%   |xi|        Lagrange multiplier
% *Output*
%   |gradP|     Gradient w.r.t. transmit power
%   |gradL|     Gradient w.r.t. the state variant Lagrange multiplier
%   |gradX|     Gradient w.r.t. Lagrange multiplier
%   |R|      	Channel capacity

%% Gradients
gradP=1./(Neff./g+P)-xi-lr*(mean(P)-Pmean);
gradL=(P-Pmax)/Pmax;
gradX=(P-Pmean)/Pmean;
R=log2(1+g.*P/Neff);

end








