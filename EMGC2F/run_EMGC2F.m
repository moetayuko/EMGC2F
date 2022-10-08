function [y_pred, obj, coeff, n, y_coar, evaltime] = run_AMAOF(As, num_clusters, use_grid)
    if nargin < 3
        use_grid = true;
    end

    if use_grid
        tic;
        [y_coar, coar_grid_cnt, As_coar] = first_nn_merge(As);
        evaltime = toc;
        n = size(As_coar{1}, 1);

        Y_init_coar = finchpp(graph_avg(As_coar), num_clusters);

        tic;
        [y_pred, obj, coeff] = emgc2f(As_coar, Y_init_coar, coar_grid_cnt);
        evaltime = evaltime + toc;
        y_pred = vec2ind(y_pred')';
        y_pred = y_pred(y_coar);
    else
        y_coar = [];
        n = size(As{1}, 1);
        Y_init = finchpp(graph_avg(As), num_clusters);
        tic;
        [y_pred, obj, coeff] = emgc2f(As, Y_init);
        evaltime = toc;
        y_pred = vec2ind(y_pred')';
    end
end
