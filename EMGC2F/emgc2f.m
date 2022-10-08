function [Y, obj, coeff] = emgc2f(As, Y, grid_cnt, max_iter)
%   Wu, D., Lu, J., Nie, F., Wang, R., & Yuan, Y. (2022). EMGC²F: Efficient
%   Multi-view Graph Clustering with Comprehensive Fusion. IJCAI.
%
%   Please contact Danyang Wu <danyangwu.cs@gmail.com> if you have any
%   questions.
%
%   SPDX-FileCopyrightText: 2022 Jitao Lu <dianlujitao@gmail.com>
%   SPDX-License-Identifier: MIT
    arguments
        As
        Y
        grid_cnt = []
        max_iter = 50
    end
    Ls = calc_laps(As);

    for iter = 1:max_iter
        view_objs = calc_view_objs(Ls, Y, grid_cnt);
        coeff = 1 ./ (2 .* view_objs);
        obj(iter) = sum(view_objs);
        if iter > 2 && abs((obj(iter) - obj(iter - 1)) / obj(iter - 1)) < 1e-9
            break;
        end

        % Y step
        L = weighted_sum(Ls, coeff);
        Y = solve_Y(L, Y, grid_cnt);
    end
end
