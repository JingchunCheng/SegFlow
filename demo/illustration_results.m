clear all; close all; clc;

addpath('../util/');

res_dir = '../results/video_SegFlow/';


images = dir(fullfile(res_dir,'*.jpg'));

for i = 1:length(images)
    im_name = images(i).name;
    im_name = im_name(1:end-4);
    img     = imread([res_dir,im_name,'.jpg']);
    load([res_dir,im_name,'.mat']);
    flo     = reshape(flo,[size(flo,2),size(flo,3),size(flo,4)]);
    flo     = permute(flo,[2,3,1]); 
    flo_img = flowToColor(flo);
     
    figure(1),
    set(gcf,'Position', [100, 100, 800,400]);
    subplot(1,2,1),hold on,title('segmentation result'),
    imshow(img),
    subplot(1,2,2),hold on,title('optical flow result'),
    imshow(flo_img),
    pause(0.2);
end


