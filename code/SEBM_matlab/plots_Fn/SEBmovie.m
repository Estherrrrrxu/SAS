function SEBmovie(elements3,coordinates,data)


v = VideoWriter('SEB.avi');
v.FrameRate = 4;
open(v); 


N = size(data,2); 
for n=1:N
    showSphere(elements3,coordinates,data(:,n), n);

    frame = getframe(gcf);
    writeVideo(v,frame);
end
close(v); 
return