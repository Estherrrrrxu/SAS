function AnimateGif(elements3,coordinates,data)

filename = 'testAnimated.gif';
h = figure;

N = size(data,2);  
for n=1:N
    u = data(:,n);
    showSphere(elements3,coordinates,data(:,n), n);
    % trisurf(elements3,coordinates(:,1),coordinates(:,2),u','facecolor','interp')
    hold on;
    view(0,90);
    
    % title('Solution of the Problem')
    % Capture the plot as an image
    frame = getframe(h);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    % Write to the GIF File
    if n == 1
        imwrite(imind,cm,filename,'gif', 'Loopcount',inf,'DelayTime',0.2);
    else
        imwrite(imind,cm,filename,'gif','WriteMode','append','DelayTime',0.2);
    end
end

return