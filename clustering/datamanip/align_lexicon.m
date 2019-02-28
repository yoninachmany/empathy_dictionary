function [ lexicon_aligned, lex_values_aligned ] = align_lexicon( lex_values, dic, names )
% Align_thesaurus - this function expects a thesaurus matrix
% the index list of words in the vocabulary, and the names
% cell array which is the new alignment and subset

%M = thes;
%n = max(size(M,1), size(M,2));
%M(n+1,n+1) = 0;
%M = triu(M);
%M = M + M';

n = length(names);
% I=1:length(dic);
% mapping=zeros(1,n);
% rev_mapping = zeros(1,(length(dic)));
% XX = zeros(n,length(dic)+1);
% for i = 1:n
%     if sum(ismember(dic, names(i))) > 0
%         XX(i,:) = M(I(ismember(dic, names(i))),:);
%         mapping(i) = I(ismember(dic, names(i)));
%         rev_mapping(I(ismember(dic, names(i)))) = i;
%     end
% end
% 
% lexicon_aligned = sparse(1,n);
% for i=1:n
%     if mapping(i)>0
%         lexicon_aligned(:,i) = XX(:,mapping(i));
%     end
% end

lex_values_aligned = zeros(1, n);
for i=1:n
    j = find(ismember(dic,names(i)));
    if j>0
        lex_values_aligned(i) = mean(lex_values(j));
    end
end
lex_values_aligned = sparse(lex_values_aligned);
lexicon_aligned = lex_values_aligned' * lex_values_aligned;
end

