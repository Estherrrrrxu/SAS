% print to pdf as the figure size, instead of Page size 
function myprintPDF2(figname,resolution)
% suggest default use: myprintPDF(figname) 
% Provide resolution only when needed.
h = gcf;
set(h,'Units','Inches');
pos = get(h,'Position'); % [left bottom width height]
% PNAS column width = 8.7cm size = [5, 4];
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4))
if exist('resolution','var')  % '-r600'
    print(h,figname,'-dpdf',resolution); % for many trajectory color plots; use when needed
else
    print(h,figname,'-dpdf');  % for color plots, scatter, bar  -- good for most
end
%{
% Set the tick fonts here: 
ftsz = 12;                         % good for figure in 0.5 shrink in paper
fgca = gca; fgca.FontSize = ftsz;  % set the labels, ticks
%}
return
