%% Some introduction
clear
clc
fprintf('Spam classification using Naive Bayes on the Ling-Spam dataset\n');
fprintf('Written by Dang Manh Truong (dangmanhtruong@gmail.com\n\n');

%% Train and test
[denominator1, denominator0, gammaY, gammaK_Y1, gammaK_Y0] = SpamClassifier_Train();
[Num_of_misclassified, Accuracy] = SpamClassifier_Test(gammaY, gammaK_Y1, gammaK_Y0);

%% Print the result
fprintf('The algorithm misclassified %d documents in the test set \n', Num_of_misclassified);
fprintf('Accuracy: %f%% \n', Accuracy);
clear
