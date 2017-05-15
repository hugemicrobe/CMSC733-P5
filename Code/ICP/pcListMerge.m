function pcOut = pcListMerge(pcList, tfList, mseList)

n = numel(pcList);
mergeSize = 0.015;
accumTform = affine3d();
pcOut = pcList{1};

for i = 2:n   
    pcCurrent = pcList{i};
    accumTform = affine3d(tfList{i-1}.T * accumTform.T);
    
    if mseList(i-1) < 2
        ptCloudAligned = pctransform(pcCurrent, accumTform);
        pcOut = pcmerge(pcOut, ptCloudAligned, mergeSize);
    end    
    if mod(i, 10) == 0
        fprintf('%d merged...\n', i);
    end    
end 

end