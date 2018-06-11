clear

cam = webcam('FaceTime HD Camera');

faceDetector = vision.CascadeObjectDetector();

figure,

k=21;
while (k<=25)
    faceImg = imread(strcat('/Users/mac/Downloads/Images/',int2str(k-20),'.jpg'));
%     faceImg=snapshot(cam);
    imshow(faceImg);
%     f = waitforbuttonpress;

%     if (f > 0)
        bbox= step(faceDetector, faceImg);
        if (size(bbox,1)== 1)
            img= imcrop(faceImg,bbox);
            img = imresize(img,[256 256]);
            imwrite(img,strcat('/Users/mac/Documents/MATLAB/Dataset/Abduvohid/',int2str(k),'i.jpg'));
            k=k+1;
        end
%     end
end
