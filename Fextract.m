img=imread('/Users/mac/Documents/MATLAB/Dataset/Abduvohid/1.jpg');
img = rgb2gray(img);
imshow(img);
[r,c,p] = size(img)

LbpWeights = [32, 64 , 128;
              16, 0 , 1;
              8, 4 , 2];
thresholdWindow = zeros(3,3);
imgLbp = zeros(r,c);
for i = 2: r - 1
    for j = 2 : c - 1
        thresholdWindow = zeros(3,3);
        
        if (img(i,j) <= img(i-1,j-1))
            thresholdWindow(1,1) = 1;
        else
            thresholdWindow(1,1) = 0;
        end
  
        if (img(i,j) <= img(i-1,j))
            thresholdWindow(1,1) = 1;
        else
            thresholdWindow(1,1) = 0;
        end
        
        if (img(i,j) <= img(i-1,j+1))
            thresholdWindow(1,1) = 1;
        else
            thresholdWindow(1,1) = 0;
        end
        
        if (img(i,j) <= img(i,j-1))
            thresholdWindow(1,1) = 1;
        else
            thresholdWindow(1,1) = 0;
        end
        
        if (img(i,j) <= img(i,j+1))
            thresholdWindow(1,1) = 1;
        else
            thresholdWindow(1,1) = 0;
        end
        
        if (img(i,j) <= img(i+1,j-1))
            thresholdWindow(1,1) = 1;
        else
            thresholdWindow(1,1) = 0;
        end
        
        if (img(i,j) <= img(i+1,j))
            thresholdWindow(1,1) = 1;
        else
            thresholdWindow(1,1) = 0;
        end
        
        if (img(i,j) <= img(i+1,j+1))
            thresholdWindow(1,1) = 1;
        else
            thresholdWindow(1,1) = 0;
        end
        
        
         imgLbp(i,j) = sum(sum(thresholdWindow.*LbpWeights));
    end
end


imshow(imgLbp);

[lbpFeatures] = extractLBPFeatures(img);