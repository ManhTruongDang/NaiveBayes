function [denominator1, denominator0, gammaY, gammaK_Y1, gammaK_Y0] = SpamClassifier_Train()
% http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex6/ex6.html
% Training phase for Naive Bayes
% The variables are named according to the equations 
% Written by Dang Manh Truong (dangmanhtruong@gmail.com)

    % Load the features
    numTrainDocs = 700;
    numTokens = 2500;
    M = dlmread('train-features.txt', ' ');
    
    spmatrix = sparse(M(:,1), M(:,2), M(:,3), numTrainDocs, numTokens);
    train_matrix = full(spmatrix);
    
    % Load the labels 
    train_labels = dlmread('train-labels.txt');
    
    % Now it's show time
    V = numTokens;
    m = numTrainDocs;
    n = sum(train_matrix , 2); % The ith-email contains n(i) words
    y = train_labels;
       
    denominator1 = y' * n + V;
    denominator0 = (1-y)' * n + V;
    
    gammaY = sum(train_labels) / m;     
    gammaK_Y1 = (sum(train_matrix(find(train_labels == 1)',:),1) + 1 ) / denominator1 ;
    gammaK_Y0 = (sum(train_matrix(find(train_labels == 0)',:),1) + 1 ) / denominator0 ;
    
    
    