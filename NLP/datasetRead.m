%'G:\Sem 8\Soft Computing\NLP\Movie Reviews\pos\*.txt');
clc;
dd = dir('F:\Sem 8\SC\NLP\Movie Reviews\pos\*.txt');
pos = {dd.name};
dd = dir('F:\Sem 8\SC\NLP\Movie Reviews\neg\*.txt');
neg = {dd.name};

        % dataset is an array of strings. it contains the contents of all the
        % file after stop words removal and stemming
        
dataset = {};
classList = zeros(length(pos)+length(neg),1);

for ii = 1:length(pos)
    filename = pos{ii};
    fid = fopen(strcat('F:\Sem 8\SC\NLP\Movie Reviews\pos\',filename));
    st = '';
    while ~feof(fid)
      a = fgetl(fid);
      st = strcat(st,a);
    end
    punctuation=regexprep(st,'[`~!@#$%^&*()[]"-_=+{}\|;:\''<,>.?/]','');
    lowercase = lower(punctuation);
    stopWordsRemoved = removeStopWords(lowercase);
    %stemmedWords = porterStemmer(stopWordsRemoved);
    stopWordsRemoved = regexprep(stopWordsRemoved,' +',' ');
    dataset{ii} = strsplit(stopWordsRemoved,' ');
    classList(ii)=1;
    fclose(fid);
end
ll = length(dataset);
for ii = 1:length(neg)
    filename = neg{ii};
    fid = fopen(strcat('F:\Sem 8\SC\NLP\Movie Reviews\neg\',filename));
    st = '';
    while ~feof(fid)
      a = fgetl(fid);
      st = strcat(st,a);
    end
    punctuation=regexprep(st,'[`~!@#$%^&*()[]"-_=+{}\|;:\''<,>.?/]','');
    lowercase = lower(punctuation);
    stopWordsRemoved = removeStopWords(lowercase);
    %stemmedWords = porterStemmer(stopWordsRemoved);
    stopWordsRemoved = regexprep(stopWordsRemoved,' +',' ');
    dataset{ll+ii} = strsplit(stopWordsRemoved,' ');
    fclose(fid);
end

stemmedWords = dataset;
    % run the porterStemmer algo to get the base word for each word
    for i = 1:length(dataset)
        for j = 1:length(dataset{i})
            if(length(dataset{i}{j})>2)
                stemmedWords{i}{j} = porterStemmer(dataset{i}{j});
            end
        end
     %   stemmedWords = horzcat(stemmedWords,s);
    end
    
    %  display(stemmedWords(i))  
   %{  
   %% step 2. TF-IDF for each word in the doc
out=zeros(size(D));
for k=1:numel(A)
  idx=ismember(D,A(k));
  out(:,k)=sum(idx,2);
end
disp(out) 

% find the unique words
% find the number of times each word occurs
% find the number of documents that contain the word
for i=1:length(stemmedWords)
    xx = stemmedWords{i};
    len = length(stemmedWords{i});
    a=unique(xx,'stable');
    b=cellfun(@(x) sum(ismember(xx,x)),a,'un',0)
    
end
%}


uwords = {};
uVar =1;
for i=1:length(stemmedWords)
    u = unique(stemmedWords{i});
    for j=1:length(u)
        uwords(uVar) = u(j);
        uVar = uVar+1;
    end
end
uniqueWords = unique(uwords);
tbl = cell2table(uniqueWords);
writetable(tbl,'F:\Sem 8\SC\NLP\Movie Reviews\uniquewords.txt');

        % to find tf, create a matrix 1000*noOfUniqueWords and count in
        % each file
nrows = length(dataset);
ncols = length(uniqueWords);
countWords = zeros(ncols);
countFiles = zeros(nrows,ncols);
    %does pattern match
    %indexC = strfind(uniqueWords,a)
     %   Index = find(not(cellfun('isempty', indexC)))
     % the count of word in each file
for i=1:nrows
    for j=1:length(stemmedWords{i})
        a=stemmedWords{i}{j};
        idx = find(strcmp([uniqueWords], a))
      %  countWords(idx) = countWords(idx)+1;
        countFiles(i,idx) = countFiles(i,idx)+1;
    end
end
% before tf-idf
dlmwrite('F:\Sem 8\SC\NLP\Movie Reviews\file1.txt', countFiles);

    % update the term frequency
    k=1;
for i=1:nrows
    l = length(stemmedWords{i})
    nz = sum(countFiles(:,k)~=0);  %finds the number of files containing that word
    idf = log(nrows/nz);
    for j =1:ncols
        countFiles(i,j) = idf*(countFiles(i,j)/l);
    end
    k=k+1;
end
dlmwrite('F:\Sem 8\SC\NLP\Movie Reviews\tfidf.txt', countFiles);
       % run info gain. inputs are each col (for each word), the class
        % vector ,no.of pos and neg class instances
posInstances = length(pos);
negInstances = length(neg);
infoGainVal = zeros(ncols,1);
for i=1:ncols
    infoGainVal(i) = infoGain(countFiles(:,i),classList,posInstances,negInstances);    
end
  
dlmwrite('F:\Sem 8\SC\NLP\Movie Reviews\ig.txt', infoGainVal);

 %sum(infoGainVal(:)~=0)=11855  nnon-zero rows
 [sorted,sortedIndex] = sort(infoGainVal,'descend');
 threshold = 15000;
 inputMatrix(:,1:15000) = countFiles(:,1:15000);
 %inputMatrix = zeros(nrows,threshold);
 for i=1:threshold
     col = sortedIndex(i);
     inputMatrix(:,i) = countFiles(:,col);
 end
 dlmwrite('F:\Sem 8\SC\NLP\Movie Reviews\inputMatrix.txt', inputMatrix);

 % calling bpn
 training_Ip = inputMatrix(1:800,1:1000);
 test_Ip= inputMatrix(801:1000,1:1000);
 training_Op = classList(1:800);
 test_Op= classList(801:1000);
 
 ll= 801
 mm = 1000;
for i=1:800
    training_Ip(ll,1:1000) = inputMatrix(i+mm,1:1000);
    training_Op(ll) = classList(i+mm);
    ll = ll+1;
end
ll=201
mm=1800
for i=1:200
    test_Ip(ll,1:1000) = inputMatrix(i+mm,1:1000);
    test_Op(ll) = classList(i+mm);
    ll = ll+1
end


classPr = bpn(inputMatrix,classList,training_Ip,training_Op,test_Ip,test_Op);