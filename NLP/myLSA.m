classdef myLSA < handle
    %MYLSA provides simple Latent Semantic Analysis methods
    %
    %   To get started, create a myLSA object this way. 
    %
    %   LSA = myLSA();
    %   
    %   You can also pass optional parameters such as delimiters,
    %   stopwords, and function hanle to a stemmer if you do not wish to
    %   use the default values. 
    %   
    %   Once initialized, myLSA object will provide number of useful
    %   methods to carry out Latent Semantic Analsysis. 
    %   
    %   Some example code:
    %   
    %       LSA = myLSA();
    %       [~,tf] = LSA.vectorize(strarray,2);
    %       Dk = LSA.lowrank(tf,0.6);
    %       Dk_norm = LSA.normalize(Dk);
    %       q_tfidf = LSA.query(query);
    %       q_reduced = LSA.reduce(q_tfidf);
    %       q_norm = LSA.normalize(q_reduced);
    %       cosine_similarity_scores = LSA.score(Dk_norm,q_norm);
    % 
    %   This class was developed and tested on MATLAB R2014b. 
    
    properties
         delimiters; % Cell array of characters as tokenization delimiters
         stopwords; % Cell array of stopwords to drop
         stemmer; % Function handle for a stemmer
         vocab; % Cell array of words used as features
         idf; % Vector of idf weights for vocab
         svd_comp; % Structure array of SVD components
    end
    
    methods
        function self = myLSA(delims, stopwords, stemmer)
            % Construct an instnace of myLSA class and populate properties
            % with default values unless overriddenn by input arguments.
            
            % sources for stopwords and Porter Stemmer
            stopwords_url = 'http://www.textfixer.com/resources/common-english-words.txt';
            stemmer_url = 'http://tartarus.org/martin/PorterStemmer/matlab.txt';
            % initialize properties based on input arguments
            if nargin == 0
                self.delimiters = {' ','.',',','-'};
                self.stopwords = self.get_stopwords(stopwords_url);
                self.stemmer = self.get_porterstemmer(stemmer_url);
            elseif nargin == 1
                self.delimiters=delims;
                self.stopwords = self.get_stopwords(stopwords_url);
                self.stemmer = self.get_porterstemmer(stemmer_url);
            elseif nargin == 2
                self.delimiters = delims;
                self.stopwords = stopwords;
                self.stemmer = self.get_porterstemmer(stemmer_url);
            else
                self.delimiters = delims;
                self.stopwords = stopwords;
                self.stemmer = stemmer;
            end
        end
        
        function [tfidf,varargout] = vectorize(self,strarray,minFreq)
            % Process a cell array of strings into a weighted document-term 
            % frequency matrix
            
            % tokenize a cell array of strings
            tokenized = self.tokenizer(strarray);
            % index the tokenized documents
            [word_lists,word_counts] = self.indexer(tokenized);
            % create a document-term frequency matrix
            docterm = self.docterm(word_lists,word_counts,minFreq);
            % apply ifidf weighting
            [tfidf, tf] = self.tfidf(docterm);
            varargout = {tf};
        end

        function weighted_q = query(self,query)
            % Process query into a vector of tfidf-weighted values
            if isempty(self.vocab)
                error('No vocabulary data in myLSA object.')
            end
            if size(query,1) ~= 1
                error('Enter a vector of char or cell')
            end
            % text pre-processing steps
            tokenized = self.tokenizer(query);
            [words,count] = self.indexer(tokenized);
            % drop any words not included in vocab
            count{1}(~ismember(words{1},self.vocab)) = [];
            words{1}(~ismember(words{1},self.vocab)) = [];
            % create a document-term frequency vector
            docterm = self.docterm(words, count);
            % apply tfidf weighting
            weighted_q = self.tfidf(docterm);
        end
        
        function q_reduced = reduce(self,weighted_q)
            % Transform query to a low rank approximation
            if isempty(self.svd_comp)
                error('No SVD components in myLSA object.')
            else
                % used the Tk and Sk from Low Rank approximation step to
                % reduce the new query into the same vector space
                q_reduced = weighted_q * self.svd_comp.Tk * self_comp.svd.Sk^-1;
            end
        end
        
        function tokenized = tokenizer(self,strarray)
            % Tokenize the text in a cell array of strings using
            % delimiters, stemmer, and stopwords defined in the properties
            
            % if it is a single line of text, place it in a cell
            if ischar(strarray) && size(strarray,1) == 1
                strarray = {strarray};
            end
            % standardize to lowercase
            strarray = lower(strarray);
            % remove numbers
            strarray = regexprep(strarray, '[0-9]+','');
            % tokenize text by delimiters
            tokenize = @(x) textscan(x, '%s', 'Delimiter', self.delimiters);
            tokenized = cellfun(tokenize, strarray);
            % drop empty cells
            dropEmpty = @(x) x(~cellfun(@isempty, x));
            tokenized = cellfun(dropEmpty, tokenized,'UniformOutput', false);
            for i = 1:length(tokenized)
                % remove stopwords
                tokenized{i}(ismember(tokenized{i},self.stopwords)) = [];
				% stem words
                tokenized{i} = cellfun(self.stemmer, tokenized{i}, 'UniformOutput', false);
            end
        end
        
        function docterm = docterm(self,words,counts,minFreq)
            % Create a document-term frequency matrix from word lists and
            % word count vectors
            
            % if minFreq is not given use all the words
            if nargin == 3
                minFreq = 1;
            end
            % if the vocab property is empty
            if isempty(self.vocab)
                % set it to the unique words from all documents
                self.vocab = unique([words{:}]);
            end
            % initialize docterm where rows = doc, cols = words
            docterm = zeros(length(words),length(self.vocab));
            % populate docterm with word count for each doc
            for i = 1:length(words)
                cols = ismember(self.vocab,words{i});
                docterm(i,cols) = counts{i};
            end
            % drop any words that didn't meet minFreq criterion
            self.vocab(sum(docterm) < minFreq) = [];
            docterm(:,sum(docterm) < minFreq) = [];
        end
        
        function [tfidf,varargout] = tfidf(self,docterm)
            % Convert raw term frequency values to weighted values using
            % TF-IDF method
            
            % compute the term frequencies for each doc
            tf = self.termfreq(docterm);
            % if idf property is empty, compute it
            if isempty(self.idf)
                self.idf = log(size(docterm,1) ./ sum(docterm));
            end
            % tfidf is a product of tf and idf
            tfidf = bsxfun(@times, tf, self.idf);
            varargout = {tf};
        end
        
        function [Dk,varargout] = lowrank(self,matrix,criteria)
            % Create a low rank approximation from sigular value
            % decomposition based on a given criteria
            
            % compute SVD
            [U,S,V] = svd(matrix);
            % if criteria is not given
            if nargin == 2
                % use the number of documents as k
                k = size(matrix,1);
            else
                % if criteria is an integer
                if floor(criteria) == criteria
                    % use it as k
                    k = criteria;
                % otherwise ctriteria is a float
                else
                    % compute the cumulative % of variance explained
                    explained = cumsum(S.^2/sum(S.^2));
                    % count the number of columns that meet the criteria
                    k = sum(explained < criteria) + 1;
                end
            end
            % create a rank-k approximation
            Dk = U(:,1:k); Sk = S(1:k,1:k); Tk = V(:,1:k);
            self.svd_comp = struct('Dk',Dk,'Sk',Sk,'Tk',Tk,'k',k);
            varargout = {Sk,Tk,k};
        end
    end
    
    methods (Static)
        function [words,counts] = indexer(tokenized)
            % Get word count vectors from a tokenized nested cell arrays of
            % strings
            
            % set up accumulators
            words = cell(length(tokenized),1);
            counts = cell(length(tokenized),1);
            % loop over each document
            for i = 1:length(tokenized)
                % remove duplicates
                [uniqs, ~, idx] = unique(tokenized{i});
                % get the word count
                count = accumarray(idx,1);         
                % update the accumulators
                words{i} = uniqs';
                counts{i} = count';
            end
        end
        
        function tf = termfreq(docterm)
            % Compute log term frequency metric
            tf = 1 + log(docterm);
            tf(tf == -Inf) = 0;
        end
        
        function normalized = normalize(matrix)
            % Length normalize each rows of a sparse matrix  
            normalized = bsxfun(@rdivide, matrix, sqrt(sum(matrix.^2,2)));
        end
        
        function sim = score(matrix,query)
            % Get cosine similarity scores from a length normalized matrix
            sim = dot(matrix,repmat(query,size(matrix,1),1),2);
        end
        
        function stopwords = get_stopwords(url)
            % Download stopwords from the web
            if exist('stopwords.txt', 'file') ~= 2
                websave('stopwords.txt',url)
            end
			stopwords = fileread('stopwords.txt');
			stopwords = strsplit(stopwords,',');
            fid = fopen('stopwords.txt');
            stopwords = textscan(fid,'%s','Delimiter',',');
            fclose(fid);
            stopwords = stopwords{:}';
        end
        
        function stemmer = get_porterstemmer(url)
            % Download Porter Stemmer from the web
            if exist('porterStemmer.m', 'file') ~= 2
                websave('porterStemmer.txt',url);
                movefile('porterStemmer.txt','porterStemmer.m')
            end
            stemmer = @porterStemmer;
        end
    end
end

