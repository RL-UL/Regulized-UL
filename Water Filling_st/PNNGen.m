function NN=PNNGen(dimIput,dimOutput,numLayers)
%% PNNGen
% _ChanGingSuny_ 2019-01-16
% Recent Update: 2019-06-18 v1.3
% 
% Generate a neural network
% 
% *Input*
%   |dimIput|	Dimension of the input
%   |dimOutput|	Dimension of the output
%   |numLayers|	Number of layers
% *Input*
%   |NN|        Neural network

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
