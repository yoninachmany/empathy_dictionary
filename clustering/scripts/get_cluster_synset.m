function [synset] = get_cluster_synset(word, XXc, names, I , J)


%word = user_word_prompt(names);

word_index = J(find(ismember(names, word)));
local_word_index = J(I==word_index);
cluster_num = J(XXc(local_word_index,:)>0)
synset = sort(names(I(XXc(:,cluster_num) >0)));
%for i =1:length(synset)
%    synset(i)
%    lex_values(ismember(lexdic,synset(i)))
%end

%msgbox(synset);

%pause;

%  allHandle = allchild(0);
%  allTag = get(allHandle, 'Tag');
%  isMsgbox = strncmp(allTag, 'Msgbox_', 7);
%  delete(allHandle(isMsgbox));
end