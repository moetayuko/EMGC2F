function [y, cnt, coarsened]= first_nn_merge(As)
    num_views = numel(As);
    n = size(As{1}, 1);

    As = cellfun(@(A) spdiags(sparse(n, 1), 0, A), As, 'UniformOutput', false);

    [~, first_nns] = cellfun(@(A) max(A, [], 2), As, 'UniformOutput', false);

    G_first_nns = cellfun(@(first_nn) sparse(1:n, first_nn, 1, n, n), ...
            first_nns, 'UniformOutput', false);
    G_shared_first_nn = sparse(n, n);
    for v = 1:num_views
        G_shared_first_nn = G_shared_first_nn + G_first_nns{v};
    end
    G_shared_first_nn = G_shared_first_nn >= fix(num_views / 2) + 1;
    G_shared_first_nn = G_shared_first_nn + G_shared_first_nn';

    [~, y] = graphconncomp(G_shared_first_nn, 'Directed', false);

    Y = ind2vec(y);
    cnt = full(sum(Y, 2));
    coarsened = cellfun(@(A) Y * A * Y', As, 'UniformOutput', false);

    y = y';
end
