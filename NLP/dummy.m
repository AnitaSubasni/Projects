A = {'dog','cat','mouse'};
D = {'dog','cat','cat'; 'cat','mouse','mouse'};
%{
out=zeros(size(D));
for k=1:numel(A)
  idx=ismember(D,A(k));
  out(:,k)=sum(idx,2);
end
disp(out)
%}

%LSA = myLSA();
s1='Working at a coffee shop adventures tacos medical school. Feminism going ';
s2='to the gym strong and confident Family Guy listening to music, my beard ';
s3='Kurosawa discussing politics trying different restaurants I know I listed ';
s4='more than 6 things. Snowboarding no drama outdoor activities discussing ';
s5='politics pickles my friends tell me they dont get why I am single.';
str = strcat(s1,s2,s3,s4,s5);
%tokenized = LSA.tokenizer(str);

%after = {strjoin(str{1},' ')}
porterStemmer(str)