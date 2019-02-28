function [IDX, XXc, J, I] = create_clusters(lex_values, lexdic, names, ...
    Embedding, ...
    sigma, thresh, K)
    % Function to create clusters from lexicon.
    %    sigma = 9.0;
    %    thresh = 0.04;
    %    K= 500;

    % figure;
    % imagesc(lex_values*lex_values');

    lex_matrix = align_lexicon(lex_values, lexdic, names);
    [Wemb] = create_W_from_embedding(sigma, thresh, [ 0.01 1 1], ...
        Embedding, lex_matrix);
    %[Wemb] = create_W_from_embedding(sigma, thresh, [1 0 0], ...
    %    Embedding); 
    D = sum(abs(Wemb));
    W = Wemb(D>0, D>0);
    [IDX, XXc] = sncut(W,K);
    %IDX = 0;
    %XXc = SpectralClustering(W,K,2);
    J=1:length(names);
    I=J(D>0);
end