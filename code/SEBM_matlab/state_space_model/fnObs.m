function  [F ,regionsEleT] = fnObs(U,t0t1,regions,elements3, TkVol)
% Observation function --- Updates from previous version
%    1. Time intervals ordered, non-overlap
%    2. multiple regions for each time interval 
%    3. no derivative (DF) returned
% The function is an integral in region and from time t0 to t1 
%       fn = \sum_{t=t0}^{t1} \sum_{Tk\in region} U_j(t) <eta_j>_{Tk}
% where <eta_j>_{Tk} is integral of eta_j in the triangle Tk
% Input
%    U       - the coefficients in the FEM
%    t0t1    - start and end time of each region       tNx2
%    regions - regions in terms of element #s;  size =[tN*rN, eN]
%              + rN = # of regions; --- assumed to the same as different t
%              + eN = maximum # of elements in all regions
%                (Regions can have different number of elements, and add 
%                 zeros if <eN)  
%    eta_all - all integral of eta_k on their elements:  NxNe sparse matrix
% Output
%    F     - the value of the function
%    DF    - derivative of the function wrt to U
% 
% Last updated by Fei Lu, 2018/6/26

tN = length(t0t1(:,1));     % t0t1 are the index number; not t_i  = i*dt
[tNrN,~]= size(regions);   

rN          = tNrN/tN; 
F           = zeros(tN,rN);
regionsEleT = zeros(tN,rN,size(regions,2)); % elements in regions at time t's
for tt =1:tN
    for rr =1:rN
        regionInd = rN*(tt-1)+rr;      %  
        ele  = nonzeros(regions(regionInd,:)); % elements in region rr at tt 
        tindx    = t0t1(tt,1)+1:1:t0t1(tt,2);
        intTkU= zeros(length(ele),1);
        for k  =1:length(ele)
            nodesEk = elements3(ele(k),:);
            intTkU(k)  = sum( sum(U(nodesEk,tindx)) )*TkVol(k);
        end
        F(tt,rr) = sum(intTkU)/(sum(3*TkVol(ele)) * (t0t1(tt,2)-t0t1(tt,1) ) );
                   % is different from note: average over space-time.
        regionsEleT(tt,rr,:) = ele ;
    end
end

return