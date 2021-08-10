function pdfseq4animate(elements3,coordinates,data)
 
figure; 
N = size(data,2);   k=1; 
for n=1:N
    u = data(:,n);clf;
    showSphere(elements3,coordinates,data(:,n), n);
    hold on;
    view(0,90);
    % tightfig;
    if 1== mod(n,5)
        figname  = strcat('fig',num2str(k)); 
        print(figname,'-dpng');
        k=k+1;
    end 
end

return