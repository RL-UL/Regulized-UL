function loss=NNTest(NN,SmpTest,LblTest)
%% Loss
y=ForwardProp(NN,reglr(SmpTest));
err=y-LblTest;
loss=sqrt(sum(err(:).*err(:)));
% loss=sqrt(sum(err(:).*err(:))/sum(LblTest(:).*LblTest(:)));
end
