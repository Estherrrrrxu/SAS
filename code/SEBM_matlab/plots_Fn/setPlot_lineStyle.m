%% plot settings: line styles
% with the linestyle, corlorArray, lineW specified in the main code
h_axes = gca; h_plot = get(h_axes,'Children');
set(h_plot,{'LineStyle'},linestyle, {'Color'}, colorArray,{'LineWidth'},lineW); 

% P = plot(rand(4));
% NameArray = {'LineStyle'};
% ValueArray = {'-','--',':','-.'}';
% set(P,NameArray,ValueArray)

