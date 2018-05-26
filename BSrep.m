classdef BSrep < handle
    properties (SetAccess = private)
        rep = 0;
        dev1 = 0;
        dev2 = 0
    end
    methods
        function BS = BSrep(I)
            % Martin:
            addpath('C:\Program Files\MATLAB\R2017a\toolbox\bsarray');
            % Pola:
            %addpath('/Users/polaschwoebel/Documents/MATLAB/bsarray');
            BS.rep = bsarray(I);
            BS.dev1 = bspartial(BS.rep, 1);
            BS.dev2 = bspartial(BS.rep, 2);
        end
        
        function interpolation = eval_fun(BS, x, y, imres)
            x(x>imres) = 0;
            y(y>imres) = 0;
            interpolation = interp2(BS.rep, x, y, 0);
        end
        
        function dev_interpolation = eval_dev1(BS, x, y, imres)
            % hacky solution to the padding problem
            x(x>imres) = 0;
            y(y>imres) = 0;
            dev_interpolation = interp2(BS.dev1, x, y, 0);
        end
        
        function dev_interpolation = eval_dev2(BS, x, y, imres)
            % hacky solution to the padding problem
            x(x>imres) = 0;
            y(y>imres) = 0;
            dev_interpolation = interp2(BS.dev2, x, y, 0);
        end
    end
end

