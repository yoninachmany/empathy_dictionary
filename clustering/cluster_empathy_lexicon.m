setup_env
[lexdic, lex_values, names] = load_lexicon('/Users/jsedoc/research/empathy_dictionary/lexica/mixed_level_ffn/distress_ratings.txt');
names = lexdic';
[WordEmbedding, WordDic, Emb] = load_word_embedding('/Users/jsedoc/research/empathy_dictionary/lexica/mixed_level_ffn/fastText_words.txt',300,names,' ');
lex_values = log(lex_values) - median(log(lex_values));
[IDX, XXc, J, I] = create_clusters(lex_values, lexdic, names, Emb, 2.0, 0.04, 500);

row = {};
for i=1:500
row = [row; {max(lex_values(I(XXc(:,i)==1))), min(lex_values(I(XXc(:,i)==1))), median(lex_values(I(XXc(:,i)==1))), strjoin(names(I(XXc(:,i)==1)),',')}];
end
T = cell2table(row, 'VariableNames',{'max', 'min', 'median', 'words'});
T = sortrows(T,'median','descend');
write(T)