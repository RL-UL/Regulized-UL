function NN=PNNGen(dimIput,dimOutput,numLayers)
%% Parameters
dimIputDflt=1;
dimOutputDflt=1;
numLayersDflt=4; % ≤ªøº¬« ‰»Î≤„
wdthHidden=4;

%% Initialzations
if nargin<1 || isempty(dimIput)  , dimIput  =dimIputDflt;	end
if nargin<2 || isempty(dimOutput), dimOutput=dimOutputDflt; end
if nargin<3 || isempty(numLayers), numLayers=numLayersDflt; end
typeHidden='Sigmoid';
typeOutput='SoftPlus';

learningRate=1e-1;

%% Neural Network
Dims=[dimIput;wdthHidden*ones(numLayers-1,1);dimOutput];
Types{numLayers,1}=typeOutput;
for l=1:numLayers-1, Types{l}=typeHidden; end

NN=NeuralNet(Dims,Types,learningRate);

end
