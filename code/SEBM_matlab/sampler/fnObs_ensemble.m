function  [F] = fnObs_ensemble(UU,region1,elements3, TkVol)
% evaluate the observation function at an ensemble at ONE time interval t0:t1
%   -- used in SMC for the computation of weight/target distribution
% The observation function is an integral in region and from time t0 to t1 
%       fn = \sum_{t=t0}^{t1} \sum_{Tk\in region} U_j(t) <eta_j>_{Tk}
% where <eta_j>_{Tk} is integral of eta_j in the triangle Tk
% Input
%    U       - an ensemble of the solution in [t0,t1]:  size [Dx,Nt,Np]
%    t0t1    - start and end time of each region     1x2 
%    regions - regions at t0t1 in terms of element #s;    rNx eleN 
%              + rN = # of regions
%              + eleN = maximum # of elements in all regions
%                (Regions can have different number of elements, and add 
%                 zeros if <eleN)  
% Output
%    F     - the value of the function,   rN x Np
%       

% NILS: This function needs updating to the new observation operator structure

[~,steps, Np]= size(UU); 
[rN,~] = size(region1); 

F = zeros(rN,Np);    
for rr =1:rN                  % Evaluate obervation function in each region
    ele   = nonzeros(region1(rr,:));    % elements in region rr
    intTkU= zeros(length(ele),Np);      
    for k  =1:length(ele)
        nodesEk = elements3(ele(k),:);  % nodes of element k
        temp    = sum(UU(nodesEk,2:end,:),1); 
        %NILS: Start summing at t0+1, here: UU(nodesEk,2:end,:) -- done
        temp    = reshape(temp,[],Np);    
        intTkU(k,:)  = sum(temp, 1)*TkVol(k);
    end 
    F(rr,:)= sum(intTkU,1)/sum(3*TkVol(ele)*(steps-1)); 
    %NILS: Averaging is missing, divide by sum(3*TkVol(ele)*steps)  -- done
end

return