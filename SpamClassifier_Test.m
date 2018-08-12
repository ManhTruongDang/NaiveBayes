function [Num_of_misclassified, Accuracy] = SpamClassifier_Test(gammaY, gammaK_Y1, gammaK_Y0)
% http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex6/ex6.html
% Testing phase for Naive Bayes
% The variables are named according to the equations 
% Written by Dang Manh Truong (dangmanhtruong@gmail.com)

    % Load the features    
    M = dlmread('test-features.txt', ' ');
    spmatrix = sparse(M(:,1), M(:,2), M(:,3));
    test_matrix = full(spmatrix);
    numTestDocs = size(test_matrix,1);
    numTokens = size(test_matrix,2);
    % Load the labels 
    test_labels = dlmread('test-labels.txt');

    % Now it's show time
    logGammaY1 = log(gammaY);
    logGammaY0 = log(1-gammaY);
    % gammaK_Y1, gammaK_Y0
    logGammaK_Y1 = log(gammaK_Y1);
    logGammaK_Y0 = log(gammaK_Y0);
    Spam = logGammaY1 + test_matrix*(logGammaK_Y1');
    NotSpam = logGammaY0 + test_matrix*(logGammaK_Y0');
    
    SpamOrNot = Spam >= NotSpam;
    Num_of_misclassified = sum(SpamOrNot ~= test_labels);
    Accuracy = (1 - Num_of_misclassified / numel(test_labels)) * 100;