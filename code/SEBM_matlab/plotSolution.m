% plotSolution 

close all;  clear all  
restoredefaultpath;   addpaths; 

%% basic setttings 
settings_ALL;    % set prior, sampler, observation, and FEM   
rng(12);

%% generate data
plotON = 0 ;  saveON = 0; thetaTrue = sampleThetaTrue(prior); 
[obs,Utrue] = generateDataNodes(thetaTrue,obsPar,femPar,'datafilename',saveON,plotON);  

elements3   = femPar.elements3; 
coordinates = femPar.coordinates; 
datapath =''; 

figure; figname = 'obs_histo'; 
printON =1; 
uObsdensity = uStatFromData(obs,Utrue,figname,datapath, printON);
if printON ==0
    histogram(obs,uObsdensity.edges(1:1:end),'Normalization','probability');
    xlabel('State  variable'); ylabel('Climatological Probability');
    legend('Observations'); setPlot_fontSize;
    myprintPDF(figname);
end

figure; figname = 'solutionAll'; 
% pos1 = [0.05 0.1 0.40 0.9]; pos2 = [0.55 0.1 0.4 0.9];  % [left bottom width height]
% subplot('Position',pos1); % subplot('Position',pos2); 
set(gcf, 'Position',  [100, 1000, 1000, 400]);  % set the figure size to be 500 pixels by 400 pixels
h1 = subplot(121); 
       tt = 10; ShowNodes =1; 
       showSphere(elements3,coordinates,full(Utrue(:,tt)), tt, ShowNodes);   
h2 = subplot(122);
       plot(Utrue(:,1:100)','linewidth',1);
       xlabel('Time step n'); ylabel('u_n');  axis([0,100,0.92,1.06]);
       title('Trajectories of all 12 nodes');   setPlot_fontSize;

 pos1 = get(h1,'Position'); % position1 = getpixelposition(h1); 
 pos1(3) = 0.36; set(h1,'Position',pos1);
 cc   = get( ancestor(h1, 'axes'), 'Colorbar'); 
 set(cc,'Limits',[0.92,1.06]);
 pos1right = cc.Position(1) + cc.Position(3); 
 pos2 = get(h2,'Position');  pos2(1) = pos1right + 0.08;  set(h2,'Position',pos2);   
 tightfig; 
 myprintPDF(figname);   


%{
figure;  figname = 'solu_sphere'; 
tt = 1; ShowNodes =1; 
showSphere(elements3,coordinates,full(Utrue(:,tt)), tt, ShowNodes);  
myprintPDF(figname);     
 
figure; plot(Utrue(:,1:100)','linewidth',1); xlabel('Time step n'); ylabel('u_n')
        title('Trajectories of all 12 nodes');   setPlot_fontSize;
        figname = 'traj_all'; 
        print(figname,'-depsc');
%}
