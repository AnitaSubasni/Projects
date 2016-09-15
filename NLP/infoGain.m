function igVal = infoGain(colValues,classValues,noOfPositives,noOfNegatives) 
    n = noOfPositives+noOfNegatives; %total number of rows
    totOccurance = 0;
    colPos = 0;
    colNeg = 0;
    % Calculation entropy of the column Values
    %Finding the probability of the column occurance
    for i = 1:n
        if (colValues(i) > 0)
            totOccurance = totOccurance+1;
            if(classValues(i) == 1)
                colPos = colPos + 1; %to calculate conditional entropy
            else
                colNeg = colNeg + 1; %to calculate conditional entropy
            end
        end
    end
    prob = totOccurance/n;
    columnEntropy = -(prob * log2(prob));
    %calculating the conditional entropy of the column Values with the
    %output class
    prob = colPos/n;
    prob1 = colNeg/n;
    ce1=0;
    ce2=0;
    if(prob >0)
       ce1= prob * log2(prob); %conditional entropy 1
    end
    if(prob1 > 0)
        ce2= prob1 * log2(prob1); %conditional entropy 2
    end
    condEntropy = -(ce1 + ce2);
    igVal = columnEntropy - condEntropy;
    igVal = abs(igVal);
end