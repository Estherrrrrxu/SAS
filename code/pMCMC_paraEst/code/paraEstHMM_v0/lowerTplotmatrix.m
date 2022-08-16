function [h,ax,BigAx,hhist,pax] = lowerTplotmatrix(x,trueValue)
%lower triangle of the Scatter plot matrix, using the plotmatrix function
% Input:
%    x      - samples, size = Mxrows
%    trueValue - size = 1xrows
%   Example:
%       x = randn(50,3); y = x*[-1 2 1;2 0 1;1 -2 3;]';
%       lowerTplotmatrix(y)
% Edited by Fei Lu, 2018-5-13

sym = '.';   % Default scatter plot symbol.
 y  = x;
dohist = 1;  % do histogram plots

rows = length(trueValue); cols = rows;
if size(x,2)~= rows 
    error(message('MATLAB:plotmatrix:arraySizeMismatch'));
end


% Don't plot anything if either x or y is empty
hhist = gobjects(0);
pax = gobjects(0);
if isempty(rows) || isempty(cols)
    if nargout>0, h = gobjects(0); ax = gobjects(0); BigAx = gobjects(0); end
    return
end


% Create/find BigAx and make it invisible
BigAx = newplot();
fig = ancestor(BigAx,'figure');
hold_state = ishold(BigAx);
set(BigAx,'Visible','off','color','none')

if any(sym=='.')
    units = get(BigAx,'units');
    set(BigAx,'units','pixels');
    pos = get(BigAx,'Position');
    set(BigAx,'units',units);
    markersize = max(1,min(15,round(15*min(pos(3:4))/max(1,size(x,1))/max(rows,cols))));
else
    markersize = get(0,'DefaultLineMarkerSize');
end

% Create and plot into axes
ax = gobjects(rows,cols);
pos = get(BigAx,'Position');
BigAxUnits = get(BigAx,'Units');
width = pos(3)/cols;
height = pos(4)/rows;
space = .02; % 2 percent space between axes
pos(1:2) = pos(1:2) + space*[width height];

xlimlow = min(x);    xlimup = max(x); 


BigAxHV = get(BigAx,'HandleVisibility');
BigAxParent = get(BigAx,'Parent');
paxes = findobj(fig,'Type','axes','tag','PlotMatrixScatterAx');
for i=rows:-1:1
    for j=i:-1:1
        axPos = [pos(1)+(j-1)*width pos(2)+(rows-i)*height ...
            width*(1-space) height*(1-space)];
        findax = findaxpos(paxes, axPos);
        if isempty(findax)
            ax(i,j) = axes('Units',BigAxUnits,'Position',axPos,'HandleVisibility',BigAxHV,'parent',BigAxParent);
            set(ax(i,j),'visible','on');
        else
            ax(i,j) = findax(1);         
        end
        hh(i,j,:) = plot(x(:,j), y(:,i),sym,'parent',ax(i,j))'; hold on; 
        if j~=i
            plot(trueValue(j),trueValue(i),'k*','MarkerSize',6*markersize); hold off;
        end
        set(hh(i,j,:),'markersize',markersize);
        set(ax(i,j),'xlimmode','auto','ylimmode','auto','xgrid','off','ygrid','off')
       
       
        % get the other triangle
        ax(j,i) = ax(i,j);   hh(j,i,:) = hh(i,j,:);
    end
end
  

% Try to be smart about axes limits and labels.  Set all the limits of a
% row or column to be the same and inset the tick marks by 10 percent.
inset = .10;
for i=1:rows 
    set(ax(i,1),'ylim',[xlimlow(i) xlimup(i)]) 
    dy = diff(get(ax(i,1),'ylim'))*inset;
    set(ax(i,1:i),'ylim',[xlimlow(i)-dy xlimup(i)+dy])
    if i<=cols && i>1
        set(ax(i,2:i),'yticklabel','')
    end
end
dx = zeros(1,cols);
for j=1:cols
    set(ax(j,j),'xlim',[xlimlow(j) xlimup(j)])
    dx(j) = diff(get(ax(j,j),'xlim'))*inset;
    set(ax(j:rows,j),'xlim',[xlimlow(j)-dx(j) xlimup(j)+dx(j)])
    if j< rows 
        set(ax(1:j,j),'xticklabel','')
    end
end

set(BigAx,'XTick',get(ax(rows,1),'xtick'),'YTick',get(ax(rows,1),'ytick'), ...
    'YLim',get(ax(rows,1),'ylim'),... %help Axes make room for y-label
    'userdata',ax,'tag','PlotMatrixBigAx')
set(ax,'tag','PlotMatrixScatterAx');

if dohist % Put a histogram on the diagonal for plotmatrix(y) case
    paxes = findobj(fig,'Type','axes','tag','PlotMatrixHistAx');
    pax = gobjects(1, rows);
    for i=rows:-1:1
        axPos = get(ax(i,i),'Position');
        findax = findaxpos(paxes, axPos);
        if isempty(findax)
            axUnits = get(ax(i,i),'Units');
            histax = axes('Units',axUnits,'Position',axPos,'HandleVisibility',BigAxHV,'parent',BigAxParent);
            set(histax,'visible','on');
        else
            histax = findax(1);
        end
        hhist(i) = histogram(histax,y(:,i,:)); hold on;
        tmp      = histax.YLim; tmp(2) = tmp(2)*0.8; 
        plot([trueValue(i) trueValue(i)],tmp,'k-*'); hold off;
        set(histax,'xtick',[],'ytick',[],'xgrid','off','ygrid','off');
        set(histax,'xlim',[xlimlow(i)-dx(i) xlimup(1,i)+dx(i)])
        set(histax,'tag','PlotMatrixHistAx');
        pax(i) = histax;  % ax handles for histograms
    end
end

% Make BigAx the CurrentAxes
set(fig,'CurrentAx',BigAx)
if ~hold_state
    set(fig,'NextPlot','replacechildren')
end

% Also set Title and X/YLabel visibility to on and strings to empty
set([get(BigAx,'Title'); get(BigAx,'XLabel'); get(BigAx,'YLabel')], ...
    'String','','Visible','on')

if nargout~=0
    h = hh;
end

tightfig;
end


function findax = findaxpos(ax, axpos)
tol = eps;
findax = [];
for i = 1:length(ax)
    axipos = get(ax(i),'Position');
    diffpos = axipos - axpos;
    if (max(max(abs(diffpos))) < tol)
        findax = ax(i);
        break;
    end
end
end
