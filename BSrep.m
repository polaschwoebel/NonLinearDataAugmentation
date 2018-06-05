classdef BSrep < handle
    properties (SetAccess = private)
        rep = 0;
        dev1 = 0;
        dev2 = 0;
		dev3 = 0;
    end
    methods
        function BS = BSrep(I, dim)
            % Martin:
            %addpath('C:\Program Files\MATLAB\R2017a\toolbox\bsarray');
            % Pola:
            %ddpath('/Users/polasachwoebel/Documents/MATLAB/bsarray');
            % server:
            addpath('bsarray');
            BS.rep = bsarray(I);
            BS.dev1 = bspartial(BS.rep, 1);
            BS.dev2 = bspartial(BS.rep, 2);
			if (dim == 3)
				BS.dev3 = bspartial(BS.rep, 3);
			end
        end

        function interpolation = eval_fun2d(BS, x, y, imres)
            x(x>imres) = 0;
            y(y>imres) = 0;
            interpolation = interp2(BS.rep, x, y, 0);
        end

		function interpolation = eval_fun3d(BS, x, y, z, shape0, shape1, shape2)
            x(x>shape0) = 0;
            y(y>shape1) = 0;
			z(z>shape2) = 0;
            interpolation = interp3(BS.rep, x, y, z, 0);
        end

        function dev_interpolation = eval_dev12d(BS, x, y, imres)
            % hacky solution to the padding problem
            x(x>imres) = 0;
            y(y>imres) = 0;
            dev_interpolation = interp2(BS.dev1, x, y, 0);
        end

		function dev_interpolation = eval_dev13d(BS, x, y, z, shape0, shape1, shape2)
            % hacky solution to the padding problem
            x(x>shape0) = 0;
            y(y>shape1) = 0;
			z(z>shape2) = 0;
            dev_interpolation = interp3(BS.dev1, x, y, z, 0);
        end

        function dev_interpolation = eval_dev22d(BS, x, y, imres)
            % hacky solution to the padding problem
            x(x>imres) = 0;
            y(y>imres) = 0;
            dev_interpolation = interp2(BS.dev2, x, y, 0);
        end

		function dev_interpolation = eval_dev23d(BS, x, y, z, shape0, shape1, shape2)
            % hacky solution to the padding problem
            x(x>shape0) = 0;
            y(y>shape1) = 0;
			z(z>shape2) = 0;
            dev_interpolation = interp3(BS.dev2, x, y, z, 0);
        end

		function dev_interpolation = eval_dev33d(BS, x, y, z, shape0, shape1, shape2)
            % hacky solution to the padding problem
            x(x>shape0) = 0;
            y(y>shape1) = 0;
			z(z>shape2) = 0;
            dev_interpolation = interp3(BS.dev3, x, y, z, 0);
        end
    end
end
