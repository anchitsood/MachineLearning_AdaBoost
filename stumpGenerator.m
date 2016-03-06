%% Header
%
% Author: Anchit Sood
% Callsign: ElessarTelcontar
% License: GNU GPLv3
% Platform: Matlab


%% stumpGenerator description
%
% This function generates a new stump given some training data and the
% existing weights matrix. The algorithm followed is described in the loop
% below.


%% Function definition
function stump = stumpGenerator(dataX, dataY, Dt)
intervals = 100;
% more intervals means better resolution

rangex1 = max(dataX(:,1)) - min(dataX(:,1));
rangex2 = max(dataX(:,2)) - min(dataX(:,2));

width = (rangex1/intervals);
height = (rangex2/intervals);
% split the x1 and x2 axes into n equal parts, where n = intervals

starterx1 = min(dataX(:,1)) - (width/2);
starterx2 = min(dataX(:,2)) - (height/2);
% choose a starting point for x1 and x2 to start sweeping through the data

currepsilon = inf;
% choose a very high epsilon, because we want to replace it with the
% minimum value of epsilon we find in each run of the loop below

stump = [0,0,0,1,0,0];
% Each stump needs to be a 1-by-6 vector: 
% Stump: [a b c d e f]
%
% Here, the equation of the stump is: 
% a*x1 + b*x2 + c > 0 or a*x1 + b*x2 + c < 0 
% 
% The choice of > or < in the above equation is dictated by 'd' in the
% stump vector: if d = 1, then output the '>' equation, if d = -1, output
% the '<' equation. The total weighted error (epsilon) from this stump is
% given by the entry 'f' in the stump vector, and the value of alpha
% calculated from this epsilon is given by the entry 'e'.
%
% Remember, here, we want our stumps to only be vertical or horizontal,
% which means a = 0, b = 1 and a = 1, b = 0 are the only two
% permissible values of 'a' and 'b' in our stump equation/vector. The
% value of 'c' actually defines the exact placement of the stump on our
% grid of training points.


% Here's how the loop works: Given a weights matrix Dt, it tries to look
% for a stump which has the minimum error on the data of interest (dataX
% and dataY). We look at the data and determine the ranges of x1 and x2,
% then we split those ranges into intervals of equal width and height.
% Next, for each interval boundary i, we take four temporary stumps: two
% horizontal and two vertical. The vertical stumps classify the data into
% left and right: one of them is a 'greater than' stump, the other one is a
% 'less than' stump, meaning one of them classifies data to its right as
% positive and left to be negative, vice versa for the other one. A similar
% logic also applies to the two horizontal stumps. For each of these 4
% stumps, we calculate the weighted error upon classifying all points using
% just the one stump. Once we have the values of the 4 errors, we take the
% minimum of these errors and store this as a temporary value of epsilon.
% Next, we move on to the next iteration of the loop and repeat the steps,
% modifying the value of epsilon if we find a better stump. Hence, for each
% iteration of the loop, we test 4 temporary stumps and determine the best
% one of these. After the loop is completed, we have the temporary stump
% that gives us the minimum weighted error, and we also have the value of
% that error. We now accept this stump as a useful stump, and store it.
% Then we use the minimum weighted error to calculate alpha, and this alpha
% value is used to modify the weights matrix Dt for the next stump to be
% generated. Refer adaboost_main.m for details of how to modify the weights
% matrix. 
for i = 1:(intervals + 1)
    
    horzRightError = sum(Dt(find(((dataX(:,1) - starterx1) .* dataY) < 0)));
    horzLeftError = sum(Dt(find(((dataX(:,1) - starterx1) .* dataY) > 0)));
    vertUpError = sum(Dt(find(((dataX(:,2) - starterx2) .* dataY) < 0)));
    vertDownError = sum(Dt(find(((dataX(:,2) - starterx2) .* dataY) > 0)));
    
    if (horzRightError <= horzLeftError)
        horzError = horzRightError;
    else
        horzError = -horzLeftError;
    end
    
    if (vertUpError <= vertDownError)
        vertError = vertUpError;
    else
        vertError = -vertDownError;
    end
    
    
    if (abs(horzError) <= abs(vertError))
        if (currepsilon > abs(horzError))
            currepsilon = abs(horzError);
            stump(1) = 1;
            stump(2) = 0;
            stump(3) = starterx1;
            stump(4) = horzError/(abs(horzError));
            stump(5) = (log((1 - currepsilon)/currepsilon))/2;
            stump(6) = currepsilon;
        else
            % do nothing, we already have the best stump
        end
        
    else
        if (currepsilon > abs(vertError))
            currepsilon = abs(vertError);
            stump(1) = 0;
            stump(2) = 1;
            stump(3) = starterx2;
            stump(4) = vertError/(abs(vertError));
            stump(5) = (log((1/currepsilon) - 1))/2;
            stump(6) = currepsilon;
        else
            % do nothing, we already have the best stump
        end
    end

    starterx1 = starterx1 + width;
    starterx2 = starterx2 + height;
end

end