function supersizeme(varargin)
% supersizeme(varargin)
% given a figure or axis handle(s), change the font sizes of all text elements to make the figure 
% more presentable before you email it to your boss or throw it into a powerpoint for your 
% presentation that starts in 5 minutes. This code has also been adapted to work with TeX markup.
% OPTIONAL INPUTS 
%   h: a handle to a figure or a vector of axes handles; If the handle is to a figure; the 
%       code will search for all text in the figure. If h is missing, code uses gcf.
%   gainFactor: a number that is multiplied to the current font size of each text element. 
%       if missing, a default value is used.  The value '1' will result in no change.  To
%       double the size of all text, enter 2.  To half the size of each text, enter either 
%       0.5 or -2 (negative values are treated as 1/abs()).  Alternatively, use the string
%       '-' or '+'(default) to decrease/increase by the default value. 
% EXAMPLE
% %Create Plot
%   fh = figure; 
%   s1=subplot(2,1,1); plot(rand(1,20), '-o'); title('RandomDots'); xlabel('x axis'); ylabel('y axis');
%   text(10,.9,'Text goes here', 'fontsize', 8)
%   s2=subplot(2,1,2); plot(rand(1,200), '-o'); title('RandomDots2'); xlabel('x axis'); ylabel('y axis');
%   suptitle('Example plot'); legend('fake data')
% %SUPERSIZE OPTIONS
%   supersizeme()                               
%   supersizeme('-')        %reverses previous line    
%   supersizeme(s1)         %only affects subplot 1
%   supersizeme([s1,s2])    %only affects subplots (not legend)
%   supersizeme(fh, '+')    %affects everything in figure
%   supersizeme(2.5)
%   supersizeme(-2.5)       %same as supersizeme(1/2.5)
%   supersizeme(1.5, fh)    %order doesn't matter
%
% Requires matlab 2016b or later.
% Danz 180522

% Source: https://www.mathworks.com/matlabcentral/fileexchange/67644-supersizeme-varargin-
% Copyright (c) 2018, Adam Danz All rights reserved

% change history
% 180906 fixed error when allFontSz is not a cell array.

%% input validity
% ishandle() will return 'true' if the gainFactor is an integer that corresponds with a figure number (douh!)
% isnumeric(), however, distinguishes between object handles and numbers (whew!)
gainIdx = cellfun(@isnumeric, varargin) | cellfun(@ischar, varargin); 
% any inputs that aren't ID'd by gainIdx must be fig/axis handles.
handIdx = ~gainIdx; 

% If user didn't enter handle, get curr fig
if ~any(handIdx)
    h = gcf; 
else
    h = varargin{handIdx}; 
end

% If user didn't enter gainFactory, set default value
defaultGain = 1.1; 
if ~any(gainIdx)
    gainFactor = defaultGain; 
else
    gainFactor = varargin{gainIdx}; 
end

% If user entered a string for the gain, replace with value
if ischar(gainFactor) && strcmp(gainFactor, '+')
    gainFactor = defaultGain; 
elseif ischar(gainFactor) && strcmp(gainFactor, '-')
    gainFactor = 1/defaultGain; 
end

% If user entered a negative gainFactor, replace with reciprocal
if gainFactor < 0
    gainFactor = 1/abs(gainFactor); 
end

%% Supersize me
% Loop through all handles and identify anything that has a 'fontsize' param
for i = 1:length(h)                   
    textHand = findall(h(i), '-property','FontSize'); 
    if isempty(textHand); continue; end
    % Get fontsize of all text
    allFontSz = get(textHand, 'FontSize'); 
    if ~iscell(allFontSz)
        allFontSz = {allFontSz}; 
    end
    % FontSizes 
    for j = 1:length(textHand)
        textHand(j).FontSize = gainFactor * allFontSz{j}; 
        % if text uses TeX markup and "\fontsize{xx}" is in string, change the fontsize from within the string
        if isprop(textHand(j), 'Interpreter') && strcmp(textHand(j).Interpreter, 'tex') && any(contains(cellstr(textHand(j).String), '\fontsize{')) %see [1]
           for k = 1:length(textHand(j).String)
               currFontCommand = regexp(textHand(j).String{k}, '\\fontsize{\d+\.?\d*}', 'match'); %search allows for possible decimals
               currFontSize = regexp(currFontCommand{:}, '\d+\.?\d*', 'match'); %{str}
               textHand(j).String(k) = strrep(textHand(j).String{k}, currFontCommand, ['\fontsize{',num2str(str2double(currFontSize{:})*gainFactor),'}']); 
           end
        end
    end
end

%% Notes
%[1] The contains() function is necessary because regexp doesn't work since '.String' can be char or cell.  
%   Furthermore, cellstr() is necessary because .String can be a cell array which will break (cellstr added 180802)
