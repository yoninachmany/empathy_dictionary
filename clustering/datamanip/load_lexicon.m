function [lexdic, lex_values, names] = load_lexicon(lex_filename)
    lex_lexicon = import_lexicon(lex_filename);

    %loads names - vocab
    %load vocab;
    names = [];

    lexdic = lex_lexicon(:,1);
    lex_values = cell2mat(lex_lexicon(:,2));
    names = [names lexdic'];
    names = unique(names);
    names = sort(names);
end