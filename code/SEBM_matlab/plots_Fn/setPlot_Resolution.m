%%  plot settings: figure resolution

%% set resolution: not for eps type vector format

fig = gcf;     % gcf: Current figure handle
size = [8.7, 6]; set(gcf,'units','centimeters','position',[0 0 size]);
% set(gcf, 'Position',  [100, 1000, 500, 400]);  % set the figure size to be 500 pixels by 400 pixels




% 1 inch = 2.54 cm % Dpi is the pixel density or dots per inch. 
% 96 dpi means there are 96 pixels per inch.
% pnas 1 column = 8.7cm, a figure 600dpi full column = 600*8.7/2.54 


% https://www.mathworks.com/help/matlab/creating_plots/save-figure-at-specific-size-and-resolution.html
% Use '-r0' to save it with screen resolution.
% print(resolution,___) uses the specified resolution. Specify the
% resolution as a character vector or string containing an integer value
% preceded by -r, for example, '-r200'. Use this option with any of the
% input arguments from the previous syntaxes.


x=(1:2:10);
y=(1:2:10);
plot(x,y);
size = [8.7 7]; set(gcf,'paperunits','centimeters','paperposition',[0 0 size]);
res = 600;
print('resized','-dtiff',['-r' num2str(res)]);
%Here I have set the width and height to be 200. Resolution is set to
%300dpi. To set the properties(?paperunnits? and ?paperposition?) of the
%image, you could use the ?set? function. ?print? function is used to set
%the resolution to 300dpi.|

