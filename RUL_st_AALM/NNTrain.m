function NN=NNTrain(numItr,SmpTrain,LblTrain,NN0)
%% Parameters
lrnRt=1e0;

%% Initializations
if nargin<4 || isempty(NN0)
    dimInput =size(SmpTrain,1);
    dimOutput=size(LblTrain,1);
    NN=NNGen(dimInput,dimOutput);
else
    NN=NN0;
end

%% SGD
t=0;
while t<numItr, t=t+1;
    y=ForwardProp(NN,reglr(SmpTrain));
    % grad=y-LblTrain;
    grad=-LblTrain./y;
    BackwardProp(NN,grad,lrnRt);
end
end
