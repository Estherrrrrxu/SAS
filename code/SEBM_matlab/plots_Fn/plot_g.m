function h = plot_g(thetatrue,lowupbounds,lineStyleColor, lineW, titl)
% plot the nonlinear function g(theta,x)
 a = lowupbounds(1); b = lowupbounds(2); dx = (b-a)/25; 
x = a:dx:b; 
gvalue = nl_fn(thetatrue,x,0); 
h = plot(x,gvalue,lineStyleColor,'linewidth',lineW); hold on; 

if exist('titl','var')
    xlabel('x'); ylabel('g(x)'); 
    title('The nonlinear function g(\theta,x)' );
    % axis([0,2,-600,600]);
    print('fig_new_g','-depsc');
end
end
