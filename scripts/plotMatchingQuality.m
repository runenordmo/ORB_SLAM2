matchingDataFilenameORB = 'matchingDataOrb00_diffTransMetric.txt';
matchingDataFilenameCNN = 'matchingDataCnn00_diffTransMetric.txt';
doPlot = true;
nMaxMatches = 2000;

% Retreive the matching data
matchDataCNN = load(matchingDataFilenameCNN);
frameNumsCNN = matchDataCNN(:,1);
nMatchesCNN = matchDataCNN(:,2);
nInliersAfterEssentialMatCNN = matchDataCNN(:,3);
nInliersAfterRecoverPoseCNN = matchDataCNN(:,4);

rotationErrorDegCNN = rad2deg(matchDataCNN(:,5));
translationErrorCNN = rad2deg(matchDataCNN(:,6));

matchDataORB = load(matchingDataFilenameORB);
frameNumsORB = matchDataORB(:,1);
nMatchesORB = matchDataORB(:,2);
nInliersAfterEssentialMatORB = matchDataORB(:,3);
nInliersAfterRecoverPoseORB = matchDataORB(:,4);
rotationErrorDegORB = rad2deg(matchDataORB(:,5));
translationErrorORB = rad2deg(matchDataORB(:,6));

%Plot the matching data
if doPlot
    figure;
    plot(frameNumsCNN,nMatchesCNN);
    hold on;
    plot(frameNumsCNN,nInliersAfterRecoverPoseCNN);
    plot(frameNumsCNN,nMatchesORB);
    plot(frameNumsCNN,nInliersAfterRecoverPoseORB);
    hold off;
    ylim([0 nMaxMatches]);
    legend('nMatches (CNN)', 'nInliersAfterRecoverPose (CNN)', ...
        'nMatches (ORB)', 'nInliersAfterRecoverPose (ORB)');
    title('Number of matches and inliers for CNN and ORB')
    xlabel('Frame number')
    
    % Plot the rotation errors
    figure;
    plot(frameNumsCNN,rotationErrorDegCNN);
    hold on;
    plot(frameNumsCNN,rotationErrorDegORB);
    
    % also plot means
    rotErrorMeanCNN = mean(rotationErrorDegCNN)
    rotErrorMeanORB = mean(rotationErrorDegORB)
    hlinerot = refline([0 rotErrorMeanCNN]);
    hlinerot.Color = 'r';
    hlinerot2 = refline([0 rotErrorMeanORB]);
    hlinerot2.Color = 'g';
    
    % set names of axes and legends
    legend('CNN','ORB');
    title('Rotation error for ORB and CNN (deg)')
    xlabel('Frame number')
    ylabel('Rotation error (deg)')

    % Plot the rotation and translation errors
    figure;
    plot(frameNumsCNN,translationErrorCNN);
    hold on;
    plot(frameNumsCNN,translationErrorORB);
    hold off;
    
    % also plot means
    translationErrorMeanCNN = mean(translationErrorCNN)
    translationErrorMeanORB = mean(translationErrorORB)
    hlinetrans = refline([0 translationErrorMeanCNN]);
    hlinetrans.Color = 'r';
    hlinetrans2 = refline([0 translationErrorMeanORB]);
    hlinetrans2.Color = 'g';
    
    % set names of axes and legends
    legend('CNN','ORB');
    title('Translation error for ORB and CNN (deg)')
    xlabel('Frame number')
    ylabel('Translation error (deg)')
    
end
