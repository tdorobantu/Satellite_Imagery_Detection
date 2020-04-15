%Tudor Dorobantu Neural Computing Submission
close all; clc; clear all;

%% Load Trained Models

load SVM_Optimal_Model.mat;
load CNN_Optimal_Model.mat;

%% Load Test Data

load CNN_TEST_DATA.mat; 
load SVM_TEST_DATA.mat;
load TEST_LABELS.mat;
    
%% predict labels using SVM and CNN and record time
tic;
[labels_SVM, posteriors_SVM] = predict(SVMModelFinal, testFeatures);
time_SVM = toc;

tic
[labels_CNN, posteriors_CNN] = classify(CNNFinal, testImgsCNN);
time_CNN = toc;

%% display confusion matrix for SVM
cm_SVM = confusionchart(testLabels,labels_SVM)

cm_SVM.Title = 'SVM Confusion Matrix';
cm_SVM.RowSummary = 'row-normalized';
cm_SVM.ColumnSummary = 'column-normalized';
%% display confusion matrix for Naive Bayes
cm_CNN = confusionchart(testLabels,labels_CNN)

cm_CNN.Title = 'CNN Confusion Matrix';
cm_CNN.RowSummary = 'row-normalized';
cm_CNN.ColumnSummary = 'column-normalized';

%%
%Precision Recall Curve for SVM & CNN Optimized Models.
[X_SVM,Y_SVM,T_SVM,AUC_SVM] = perfcurve(testLabels,posteriors_SVM(:,2), 'positive',"XCrit",'reca', 'YCrit','prec');
[X_CNN,Y_CNN,T_CNN,AUC_CNN] = perfcurve(testLabels,posteriors_CNN(:,2), 'positive',"XCrit",'reca', 'YCrit','prec');

plot(X_SVM,Y_SVM);
hold on
plot(X_CNN,Y_CNN);
xlabel('Recall');
ylabel('Precision');
title('Precision-Recall Curve RF vs NB')
hold off
legend('SVM', 'CNN', 'Location', 'northwest')

%%
%ROC Curve for SVM & CNN Optimized Models.
[X_SVM,Y_SVM,T_SVM,AUC_SVM] = perfcurve(testLabels,posteriors_SVM(:,2), 'positive');
[X_CNN,Y_CNN,T_NB,AUC_CNN] = perfcurve(testLabels,posteriors_CNN(:,2), 'positive');

plot(X_SVM,Y_SVM);
hold on
plot(X_CNN,Y_CNN);
xlabel('FPR');
ylabel('TPR');
title('ROC Curve SVM vs CNN')
hold off
legend('SVM', 'CNN', 'Location', 'northeast')

%% display performance metrics and prediction time for SVM

cm = confusionmat(testLabels,labels_SVM);
eval_metrics('Performance Results for SVM', cm);
fprintf('\n auc: %.3f \n', AUC_SVM)
fprintf('predict time: %.3f seconds \n', time_SVM)

%% display performance metrics and prediction time for CNN
cm = confusionmat(testLabels,labels_CNN);
eval_metrics('Performance Results for CNN', cm);
fprintf('\n auc: %.3f \n', AUC_CNN)
fprintf('predict time: %.3f seconds \n', time_CNN)

%% function to calculate performance metrics

function [error, recall, precision, f1] = eval_metrics(modelname, cm)
    error = 1 - sum(diag(cm))/sum(cm(:));
    recall = cm(2,2)/sum(cm(2,:));
    precision = cm(2,2)/sum(cm(:,2));
    f1 = 2*recall*precision/(recall+precision);
    fprintf(['%s: \n error=%.3f, \n recall=%.3f, \n precision=%.3f, \n f1=%.3f'], modelname, error, recall, precision, f1)
end




