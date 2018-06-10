clear

cam = webcam('FaceTime HD Camera');

faceDetector = vision.CascadeObjectDetector();

figure,

k=1;
while (k<=20)
    faceImg = imread(strcat('/Users/mac/Downloads/Abduvohid/',int2str(k),'.jpg'));
    imshow(faceImg);
%     f = waitforbuttonpress;

%     if (f > 0)
        bbox= step(faceDetector, faceImg);
        if (size(bbox,1)== 1)
            img= imcrop(faceImg,bbox);
            img = imresize(img,[256 256]);
            imwrite(img,strcat('/Users/mac/Documents/MATLAB/Dataset/Abduvohid/',int2str(k),'.jpg'));
            k=k+1;
        end
%     end
end
