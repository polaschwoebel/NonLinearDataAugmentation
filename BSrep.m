classdef BSrep < handle
    properties (SetAccess = private)
        rep = 0;
        dev1 = 0;
        dev2 = 0
    end
    methods
        function BS = BSrep(I)
            addpath('C:\Program Files\MATLAB\R2017a\toolbox\bsarray');
            BS.rep = bsarray(I);
            BS.dev1 = bspartial(BS.rep, 1);
            BS.dev2 = bspartial(BS.rep, 2);
        end
        
        function interpolation = eval_fun(BS, x, y)
            interpolation = interp2(BS.rep, x, y);
        end
        
        function dev_interpolation = eval_dev1(BS, x, y)
            dev_interpolation = interp2(BS.dev1, x, y);
        end
        
        function dev_interpolation = eval_dev2(BS, x, y)
            dev_interpolation = interp2(BS.dev2, x, y);
        end
    end
end

