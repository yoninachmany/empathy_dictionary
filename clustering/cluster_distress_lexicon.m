setup_env
[lexdic, lex_values, names] = load_lexicon('/Users/jsedoc/research/empathy_dictionary/lexica/mixed_level_ffn/distress_ratings.txt');
names = lexdic';
[WordEmbedding, WordDic, Emb] = load_word_embedding('/Users/jsedoc/research/empathy_dictionary/lexica/mixed_level_ffn/fastText_words.txt',300,names,' ');
lex_values = log(lex_values) - median(log(lex_values));
[IDX, XXc, J, I] = create_clusters(lex_values, lexdic, names, Emb, 2.0, 0.04, 500, [0 1 1]);

row = {};
for i=1:500
    [~, TF] = rmoutliers(lex_values(I(XXc(:,i)==1)));
    I_T = I(XXc(:,i)==1); I_TT = I_T(TF==0);
    [~, firstsortorder] = sort(lex_values(I_TT), 'descend'); I_TT = I_TT(firstsortorder); high_to_low_names=names(I_TT); 
    [~, firstsortorder] = sort(lex_values(I_TT), 'ascend');  I_TT = I_TT(firstsortorder); low_to_high_names=names(I_TT);
row = [row; {i, max(lex_values(I_TT)), min(lex_values(I_TT)), median(lex_values(I_TT)), strjoin(high_to_low_names,','), strjoin(low_to_high_names,',')}];
end
clear TF X I_T I_TT tmp firstsordered firstsortorder
distress_clusters = cell2table(row, 'VariableNames',{'cluster_num','max', 'min', 'median', 'high_to_low_words', 'low_to_high_words'});
distress_clusters = sortrows(distress_clusters,'median','descend');
write(distress_clusters)

% row = {};
% for  i = 1:10
% word = distressratings.tokens{i} ;
% lv = lex_values(find(ismember(lexdic, word))) + 2.5;
% row = [row; {lv,word}];
% end
% listtable = cell2table(row, 'VariableNames',{'word','rating'});