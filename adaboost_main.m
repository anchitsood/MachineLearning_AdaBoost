%% Header
%
% Author: Anchit Sood
% Callsign: ElessarTelcontar
% License: GNU GPLv3
% Platform: Matlab


%% adaboost_main.m description
%
% This script is a simple hands on demonstration of the Adaptive Boosting
% algorithm. It generates a training set and a testing set of examples.
% Each of these sets is a point in the (x1, x2) space, and each has a
% label of +1 or -1 (meaning that each point can only be classified one of
% two ways). In other words, each of these sets is a set of points, and
% each point has an assciated label of either +1 or -1 (yes or no, true or
% false, spam or not spam, whichever makes more sense to the reader).
%
% Then, after generating the data sets, the script generates stumps or
% 'weak' decision boundaries using the train data. Each stump generated is 
% the best possible stump considering the weights assigned to the points
% from the previous stump (the first stump is generated from an arbitrary
% equal weight assigned to all data points).
%
% Next, the script combines the stumps according to the AdaBoost
% algorithm formula to generate a classifier which is known to be an
% extremely strong classifier.
%
% The strength of the classifier is all well and good when stated
% verbally, but we would like some metric which tests the performance of
% this classifier. So, the last step of thisscript tests the classifier on
% both the training set and the testing set. Some numbers are generated,
% comparing the performance on both datasets.


%% Parameters 

% Feel free to change the parameters for fun
% I used trainsize = 400; testsize = 100

trainsize = 400;
% number of training points to generate
testsize = 100;
% number of testing points to generate
stumps_to_generate = 20;
% number of 'weak' learners/stumps to generate
% more stumps = better classifier

[trainX, trainY] = gen_sample(trainsize);
% generate train set (check out gen_sample script)
[testX, testY] = gen_sample(testsize);
% generate test set (check out gen_sample script)


Dt = zeros(trainsize,1);
Dt = Dt + (1/trainsize);
% Dt = weights matrix (trainsize-by-1 matrix) 
%
% Dt is a matrix containing the weights (importance) assigned to each point
% in a set. The AdaBoost algorithm requires us to increase the weights of
% the points which our previous stump classified wrongly. As a starting
% point, we assign equal weights to all points in our training set, meaning
% we do not prefer one over the other. Then, after generating each decision
% stump, we check which points are classified wrongly by this stump. Now we
% give more importance to such wrong classified points, and we would like
% out next stump to do a better job of classifying these points. So, we
% increase the weights assigned to these points, and correspondingly reduce
% the weights assigned to the correctly classified points.
%
% Here's the weight update rule after generating each stump:
% Wrong point i: Dt(i) -> Dt(i)*exp(alpha)
% (This is an increase in weight)
% Correct point j: Dt(j) -> Dt(j)*exp(-alpha)
% (This is a reduction in weight)
%
% The alpha is the correction factor after testing each point with a new
% stump. It ensures that the next update of the weights matrix will be such
% that half the weight will be put on correct examples, and half on the
% incorrect ones.
%
% Finally, since we need the weights to sum to 1, we
% normalize the weights matrix, meaning we sum all its entries, and divide
% each entry by this sum. This finally gives us a new Dt to use for
% generating a new stump.

stumps = [];
% Begin with no decision stumps


%% Generate stumps and add to the stumps[] matrix
for i = 1:stumps_to_generate
    
    newStump = stumpGenerator(trainX, trainY, Dt);
    % Generate a new stump from the training data, using the weights from
    % the previous stump. Each stump generated is the best possible using
    % the current weights, because stumpGenerator checks the total error
    % arising out of every possible configuration of that stump, and then
    % chooses the one which has the least error. The function
    % stumpGenerator also returns the epsilon (total error) and alpha
    % (weight update metric) based on its chosen stump.
    %
    % Each stump generated by stumpGenerator is a 1-by-6 vector: 
    % Stump: [a b c d e f]
    %
    % Here, the equation of the stump is: 
    % a*x1 + b*x2 + c > 0 or a*x1 + b*x2 + c < 0 
    % 
    % The choice of > or < in the above equation is dictated by 'd' in the
    % stump vector: if d = 1, then output the '>' equation, if d = -1,
    % output the '<' equation. The total weighted error (epsilon) from this
    % stump is given by the entry 'f' in the stump vector, and the value of
    % alpha calculated from this epsilon is given by the entry 'e'.
    %
    % Remember, here, we want our stumps to only be vertical or horizontal,
    % which means a = 0, b = 1 and a = 1, b = 0 are the only two
    % permissible values of 'a' and 'b' in our stump equation/vector. The
    % value of 'c' actually defines the exact placement of the stump on our
    % grid of training points.
    
    
    stumps = vertcat(stumps,newStump);
    % Update the stumps matrix by vertically concatenating this newly
    % generated stump to the exiting list of stumps.
    
    
    % Provided below are the weight update rules for the 4 possible cases:
    % 
    % [Case 1]: If the stump is a vertical stump, meaning it separates
    % points into 'left' and 'right', and if this is a stump equation of
    % the form '>', then the following points are wrongly classified:
    % [Case 1](Wrongly classified points): (all points to the right of this
    % stump which have a label of -1) + (all points to the left of this
    % stump which have a label of +1)
    %
    % [Case 2]: If the stump is a vertical stump, meaning it separates
    % points into 'left' and 'right', and if this is a stump equation of
    % the form '<', then the following points are wrongly classified:
    % [Case 2](Wrongly classified points): (all points to the right of this
    % stump which have a label of +1) + (all points to the left of this
    % stump which have a label of -1)
    %
    % [Case 3]: If the stump is a horizontal stump, meaning it separates
    % points into 'up' and 'down', and if this is a stump equation of the
    % form '>', then the following points are wrongly classified:
    % [Case 3](Wrongly classified points): (all points above this stump
    % which have a label of -1) + (all points below this stump which have a
    % label of +1)
    %
    % [Case 4]: If the stump is a horizontal stump, meaning it separates
    % points into 'up' and 'down', and if this is a stump equation of the
    % form '<', then the following points are wrongly classified:
    % [Case 4](Wrongly classified points): (all points above this stump
    % which have a label of +1) + (all points below this stump which have a
    % label of -1)
    %
    % Update the weights of these points using the weight update metric,
    % then update the weights of the correctly classified points as well
    % using the same weight update metric, namely the value of alpha
    % associated with this stump (using the update rule defined above in
    % the explaination of the Dt matrix).
    if (newStump(1) == 1 && newStump(2) == 0)
        if (newStump(4) == 1)
            Dt(find(((trainX(:,1) - newStump(3)) .* trainY) < 0)) = (Dt(find(((trainX(:,1) - newStump(3)) .* trainY) < 0)))*exp(newStump(5));
            Dt(find(((trainX(:,1) - newStump(3)) .* trainY) >= 0)) = (Dt(find(((trainX(:,1) - newStump(3)) .* trainY) >= 0)))*exp(-newStump(5));
        else
            Dt(find(((trainX(:,1) - newStump(3)) .* trainY) > 0)) = (Dt(find(((trainX(:,1) - newStump(3)) .* trainY) > 0)))*exp(newStump(5));
            Dt(find(((trainX(:,1) - newStump(3)) .* trainY) <= 0)) = (Dt(find(((trainX(:,1) - newStump(3)) .* trainY) <= 0)))*exp(-newStump(5));
        end
      
    elseif (newStump(1) == 0 && newStump(2) == 1)
        if (newStump(4) == 1)
            Dt(find(((trainX(:,2) - newStump(3)) .* trainY) < 0)) = (Dt(find(((trainX(:,2) - newStump(3)) .* trainY) < 0)))*exp(newStump(5));
            Dt(find(((trainX(:,2) - newStump(3)) .* trainY) >= 0)) = (Dt(find(((trainX(:,2) - newStump(3)) .* trainY) >= 0)))*exp(-newStump(5));
        else
            Dt(find(((trainX(:,2) - newStump(3)) .* trainY) > 0)) = (Dt(find(((trainX(:,2) - newStump(3)) .* trainY) > 0)))*exp(newStump(5));
            Dt(find(((trainX(:,2) - newStump(3)) .* trainY) <= 0)) = (Dt(find(((trainX(:,2) - newStump(3)) .* trainY) <= 0)))*exp(-newStump(5));
        end
        
    else
        break;
    end
    
    Dt = Dt/sum(Dt);
    % Normalize the new weights matrix
end


%% Generate the final classifier based on the stumps

trainclassifier = zeros(trainsize,stumps_to_generate);
% this matrix stores the predicted labels of the points which are output by
% testing against our stumps: trainclassfier(i,j) referes to the presicted
% label of the point i with respect to the classifier built by prematurely
% combining all stumps from 1 through j

trainmargins = zeros(trainsize,stumps_to_generate);
% this matrix stores the margins of all points with respect to all stumps:
% testmargins(i,j) refers to the margin of the point i with respect to the
% classifier built by prematurely combining all stumps from 1 through j

mintrainmargins = zeros(1,stumps_to_generate);
% this matrix stores the minimum margins with respect to each particular
% stump: mintestmargins(1,i) takes all points in the dataset and looks at
% their margins with respect to the classifier built by prematurely
% combining all stumps from 1 through i, then selects the minimum of these
% margins and stores it
% simply, mintestmargins(1,i) = min(testmargins(:,i));


% Now that we have all our stumps, we aim to develop our final calssifier
% by combining the information from all these stumps. Here's how we go
% about doing it:
%
% We select a point from our training set, then test that point with all
% our stumps. Each stump will either output a +1 or -1 for that point,
% based on where the point is in comparoson to this stump (If this is a
% vertical stump of the '>' form, for example, it will output a +1 for a
% point which is to its right, and a -1 for a point to its left). We take
% the value of this output from each stump, multiply it with the
% corresponding alpha of that stump, and then finally add this number up
% from all the stumps for this particular point. Finally, if this sum is
% positive, we output a +1, or a -1 if the sum is negative. This output is
% the final result about that point that our strong classifier gave us.
for i = 1:trainsize
    alphasum = 0;
    tempclassifier = 0;
    for j = 1:size(stumps,1)
        if (stumps(j,1) == 1) && (stumps(j,2) == 0)
            if (stumps(j,4) == 1)
                tempclassifier = tempclassifier + (sign(trainX(i,1) - stumps(j,3)))*stumps(j,5);
            else
                tempclassifier = tempclassifier - (sign(trainX(i,1) - stumps(j,3)))*stumps(j,5);
            end
            
        elseif (stumps(j,1) == 0) && (stumps(j,2) == 1)
            if (stumps(j,4) == 1)
                tempclassifier = tempclassifier + (sign(trainX(i,2) - stumps(j,3)))*stumps(j,5);
            else
                tempclassifier = tempclassifier - (sign(trainX(i,2) - stumps(j,3)))*stumps(j,5);
            end
            
        else
            break;
        end
        alphasum = alphasum + stumps(j,5);
        trainmargins(i,j) = (trainY(i) * tempclassifier) / alphasum;
        trainclassifier(i,j) = sign(tempclassifier);
    end
end

for i = 1:stumps_to_generate
    mintrainmargins(1,i) = min(trainmargins(:,i));
end

%% Use the stumps to test the points in the testset

testclassifier = zeros(testsize,stumps_to_generate);
% this matrix stores the predicted labels of the points which are output by
% testing against our stumps: testclassfier(i,j) referes to the presicted
% label of the point i with respect to the classifier built by prematurely
% combining all stumps from 1 through j

testmargins = zeros(testsize,stumps_to_generate);
% this matrix stores the margins of all points with respect to all stumps:
% testmargins(i,j) refers to the margin of the point i with respect to the
% classifier built by prematurely combining all stumps from 1 through j

mintestmargins = zeros(1,stumps_to_generate);
% this matrix stores the minimum margins with respect to each particular
% stump: mintestmargins(1,i) takes all points in the dataset and looks at
% their margins with respect to the classifier built by prematurely
% combining all stumps from 1 through i, then selects the minimum of these
% margins and stores it
% simply, mintestmargins(1,i) = min(testmargins(:,i));


% This loop does the exact same thing as described above in generating the
% classifier, except it uses the test points data set instead of the train
% points.
for i = 1:testsize
    alphasum = 0;
    tempclassifier = 0;
    for j = 1:size(stumps,1)
        if (stumps(j,1) == 1) && (stumps(j,2) == 0)
            if (stumps(j,4) == 1)
                tempclassifier = tempclassifier + (sign(testX(i,1) - stumps(j,3)))*stumps(j,5);
            else
                tempclassifier = tempclassifier - (sign(testX(i,1) - stumps(j,3)))*stumps(j,5);
            end
            
        elseif (stumps(j,1) == 0) && (stumps(j,2) == 1)
            if (stumps(j,4) == 1)
                tempclassifier = tempclassifier + (sign(testX(i,2) - stumps(j,3)))*stumps(j,5);
            else
                tempclassifier = tempclassifier - (sign(testX(i,2) - stumps(j,3)))*stumps(j,5);
            end
            
        else
            break;
        end
        alphasum = alphasum + stumps(j,5);
        testmargins(i,j) = (testY(i) * tempclassifier) / alphasum;
        testclassifier(i,j) = sign(tempclassifier);
    end
end

for i = 1:stumps_to_generate
    mintestmargins(1,i) = min(testmargins(:,i));
end


%% Some numerical tests on the quality of the classifiers

trainaccuracy = zeros(1,stumps_to_generate);
% this vector stores the number of successes of the combined learner on the
% train data using successively increasing number of stumps:
% trainaccuracy(1,i) refers to the number of correct hits with respect to
% the classifier built by prematurely combining all stumps from 1 through i

testaccuracy = zeros(1,stumps_to_generate);
% this vector stores the number of successes of the combined learner on the
% test data using successively increasing number of stumps:
% trainaccuracy(1,i) refers to the number of correct hits with respect to
% the classifier built by prematurely combining all stumps from 1 through i

for i = 1:stumps_to_generate
    trainaccuracy(1,i) = size(find(trainclassifier(:,i) == trainY),1);
    testaccuracy(1,i) = size(find(testclassifier(:,i) == testY),1);
end

trainerror = size(trainY,1) - trainaccuracy;
% this is simply the number of incorrect hits on the train data: just
% subtract the total number of correct hits from the data size

testerror = size(testY,1) - testaccuracy;
% this is simply the number of incorrect hits on the test data: just
% subtract the total number of correct hits from the data size

trainerror_rate = trainerror/size(trainY,1);
% normalized train error

testerror_rate = testerror/size(testY,1);
% normalized test error


%% Some plots to visualize the quality of the classifiers

% plot all training data
invariant = [-2,2];
figure(1);
title('Training data: legend shows actual labels for points');
xlabel('x_1');
ylabel('x_2');
hold on;
positives = scatter(trainX(find(trainY == 1),1),trainX(find(trainY == 1),2),10,'Blue','o');
negatives = scatter(trainX(find(trainY == -1),1),trainX(find(trainY == -1),2),10,'Black','d');
legend([positives,negatives],'+1','-1','Location','NorthEast');
hold off;


% plot all stumps on training data plot
figure(2);
title('Training data with stumps: * marker shows the direction in which a stump is active');
xlabel('x_1');
ylabel('x_2');
hold on;
positives = scatter(trainX(find(trainY == 1),1),trainX(find(trainY == 1),2),10,'Blue','o');
negatives = scatter(trainX(find(trainY == -1),1),trainX(find(trainY == -1),2),10,'Black','d');
legend([positives,negatives],'+1','-1','Location','NorthEast');
horz = stumps(find(stumps(:,1) == 1),:);
vert = stumps(find(stumps(:,1) == 0),:);
horzpos = horz(find(horz(:,4) == 1),:);
horzneg = horz(find(horz(:,4) == -1),:);
vertpos = vert(find(vert(:,4) == 1),:);
vertneg = vert(find(vert(:,4) == -1),:);

for i = 1:size(horzpos,1)
    plot([horzpos(i,3),horzpos(i,3)],invariant,'Green');
    plot(horzpos(i,3)+0.02,1.95,'*','Markersize',5,'Color','Green');
end

for i = 1:size(horzneg,1)
    plot([horzneg(i,3),horzneg(i,3)],invariant,'Red');
    plot(horzneg(i,3)-0.02,1.95,'*','Markersize',5,'Color','Red');
end

for i = 1:size(vertpos,1)
    plot(invariant,[vertpos(i,3),vertpos(i,3)],'Cyan');
    plot(1.95,vertpos(i,3)+0.02,'*','Markersize',5,'Color','Cyan');
end

for i = 1:size(vertneg,1)
    plot(invariant,[vertneg(i,3),vertneg(i,3)],'Magenta');
    plot(1.95,vertneg(i,3)-0.02,'*','Markersize',5,'Color','Magenta');
end

hold off;


% plot all testing data
figure(3);
title('Testing data: legend shows actual labels for points');
xlabel('x_1');
ylabel('x_2');
hold on;
positives = scatter(testX(find(testY == 1),1),testX(find(testY == 1),2),10,'Blue','o');
negatives = scatter(testX(find(testY == -1),1),testX(find(testY == -1),2),10,'Black','d');
legend([positives,negatives],'+1','-1','Location','NorthEast');
hold off;


% plot all stumps on testing data plot
figure(4);
title('Testing data with stumps: * marker shows the direction in which a stump is active');
xlabel('x_1');
ylabel('x_2');
hold on;
positives = scatter(testX(find(testY == 1),1),testX(find(testY == 1),2),10,'Blue','o');
negatives = scatter(testX(find(testY == -1),1),testX(find(testY == -1),2),10,'Black','d');
legend([positives,negatives],'+1','-1','Location','NorthEast');
horz = stumps(find(stumps(:,1) == 1),:);
vert = stumps(find(stumps(:,1) == 0),:);
horzpos = horz(find(horz(:,4) == 1),:);
horzneg = horz(find(horz(:,4) == -1),:);
vertpos = vert(find(vert(:,4) == 1),:);
vertneg = vert(find(vert(:,4) == -1),:);

for i = 1:size(horzpos,1)
    plot([horzpos(i,3),horzpos(i,3)],invariant,'Green');
    plot(horzpos(i,3)+0.02,1.95,'*','Markersize',5,'Color','Green');
end

for i = 1:size(horzneg,1)
    plot([horzneg(i,3),horzneg(i,3)],invariant,'Red');
    plot(horzneg(i,3)-0.02,1.95,'*','Markersize',5,'Color','Red');
end

for i = 1:size(vertpos,1)
    plot(invariant,[vertpos(i,3),vertpos(i,3)],'Cyan');
    plot(1.95,vertpos(i,3)+0.02,'*','Markersize',5,'Color','Cyan');
end

for i = 1:size(vertneg,1)
    plot(invariant,[vertneg(i,3),vertneg(i,3)],'Magenta');
    plot(1.95,vertneg(i,3)-0.02,'*','Markersize',5,'Color','Magenta');
end

hold off;


% plot weighted error rate epsilon of each 'weak' learner/stump
figure(5);
plot((1:stumps_to_generate),stumps(:,6));
title('Weighted error rate \epsilon_t of the stump generated at iteration number t');
xlabel('stump number t');
ylabel('\epsilon_t');


% plot the training and testing errors (count)
figure(6);
hold on;
plot(1:stumps_to_generate,trainerror);
plot(1:stumps_to_generate,testerror);
legend('trainerrors','testerrors');
title('Number of errors based on classifier built by combining t stumps')
xlabel('stumps used t');
ylabel('number of errors');
hold off;


% plot the training and testing errors (normalized rate)
figure(7);
hold on;
plot(1:stumps_to_generate,trainerror_rate);
plot(1:stumps_to_generate,testerror_rate);
legend('normalized trainerror','normalized testerror');
title('Percentage/rate of errors based on classifier built by combining t stumps')
xlabel('stumps used t');
ylabel('rate of errors');
hold off;


% plot the minimum margins
figure(8);
% hold on;
plot(1:stumps_to_generate,mintrainmargins);
% plot(1:stumps_to_generate,mintestmargins);
% legend('minimum train margin','minimum test margin');
title('Minimum margin out of the entire training data set based on classifier built by combining t stumps')
xlabel('stumps used t');
ylabel('min margin');
% hold off;


% some useful output!
output0 = strcat({'The final combined classifier used '}, num2str(size(stumps,1)), {' weak learners/stumps.'});
output1 = strcat({'It classified '}, num2str(trainaccuracy(1,stumps_to_generate)), {' points out of '}, num2str(size(trainY,1)), {' correctly in the training set.'});
output2 = strcat({'Further, it classified '}, num2str(testaccuracy(1,stumps_to_generate)), {' points out of '}, num2str(size(testY,1)), {' correctly in the testing set.'});
disp(output0);
disp(output1);
disp(output2);


% housekeeping: delete unused variables
clear alphasum invariant i j output0 output1 output2;
% clear all;