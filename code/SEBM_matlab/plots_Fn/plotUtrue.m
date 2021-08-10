function plotUtrue(Utrue)
% plot Utrue to see if the parameters are reasonable
figure
subplot(211); plot(Utrue(:,1:100)'); title('Trajectories of all 12 nodes'); 
              setPlot_fontSize;
subplot(212); mUtrue = mean(Utrue,2);   stdUtrue = std(Utrue,0,2); 
        plot(mUtrue); hold on; plot(mUtrue+stdUtrue); plot(mUtrue-stdUtrue); 
        title('Mean and std along the trajectoris'); 
        legend('Mean \pm std'); set(legend,'Fontsize',12);
         setPlot_fontSize;      
return
