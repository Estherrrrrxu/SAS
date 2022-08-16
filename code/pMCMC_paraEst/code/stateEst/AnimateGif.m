function AnimateGif(Xarray,farray,titl)

filename = 'testAnimated.gif';
h = figure;

N = size(Xarray,1); 
for n=1:N
    x = Xarray(n,:); f= farray(n,:); 
    plot(x,f,'linewidth',1);  
    title(titl(n))
    % Capture the plot as an image
    frame = getframe(h);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    % Write to the GIF File
    if n == 1
        imwrite(imind,cm,filename,'gif', 'Loopcount',inf,'DelayTime',3);
    elseif n==N
        imwrite(imind,cm,filename,'gif','WriteMode','append','DelayTime',2);
    else 
        imwrite(imind,cm,filename,'gif','WriteMode','append','DelayTime',0.5);
    end
end

return