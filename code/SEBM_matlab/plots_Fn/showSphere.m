function showSphere(elements3,coordinates,u,tt,ShowNodes)
%SHOWSphere   Presents piecewise affine function graphically over the sphere.
% Input:     
%   ELEMENTS3   -  a set of triangles; each row: (node1, node2, node3)
%                  The nodes have to be counted clockwise or anti-clockwise. 
%   coordinates - coordinates of the nodes:  (no. of nodes) x 3  
%                   
% By Nils Weitzel and Fei Lu, 2018/3/3

trisurf(elements3,coordinates(:,1),coordinates(:,2),coordinates(:,3),u','facecolor','interp'); 
hold on
colorbar;
angle = rand(1)*180; 
view(0,angle);
if exist('ShowNodes','var') && ShowNodes==1 
% Display only Node Numbers
    hold on; nnode = length(coordinates(:,1));
    for i = 1:nnode
        text(coordinates(i,1),coordinates(i,2),coordinates(i,3),int2str(i),....
            'fontsize',12,'fontweight','bold','Color','k'); 
    end 
    xlabel('\xi_1/\pi'); ylabel('\xi_2/\pi');
    cc =get( ancestor(gca, 'axes'), 'Colorbar'); cc.FontSize = 12; 
    set(gca,'FontSize',12);
end
titleT = strcat('Solution at time step n =  ', num2str(tt, '%2g'));   title(titleT);
% tightfig; 
%     if 1== mod(tt,5)
%         figname  = strcat('fig',num2str(tt)); 
%         print(figname,'-dpdf');
%     end    
hold off;

