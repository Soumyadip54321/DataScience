function [pixelspread_newglwise,graylvl_dist,output,pixel_count]=histogram_eq(image,bits)
a=double(imread(image));
dim=size(a);
equalized_img=uint8(zeros(dim(1),dim(2)));
numpixels_intensitywise=zeros(1,2^bits);
for i=1:dim(1)
    for j=1:dim(2)
            numpixels_intensitywise(a(i,j)+1)=numpixels_intensitywise(a(i,j)+1)+1;
    end
end
output=numpixels_intensitywise;
%bar(output);
%title('Distribution of pixels intensity-wise');
%xlabel('graylevels');ylabel('number of pixels');
pixel_count=sum(output(:));
pdf=output/pixel_count;
cdf=zeros(1,2^bits);
for k=1:2^bits
    cdf(k)=sum(pdf(1:k));
end
bar(pdf,'r');
title('pdf of pixel distribution');
xlabel('graylevels');ylabel('probability of pixels');
figure,bar(cdf,'b');
title('cdf of pixel distribution');
xlabel('graylevels');ylabel('cdf of pixels');
graylvl_dist=round((2^bits-1)*cdf);
pixelspread_newglwise=zeros(1,2^bits);
for i=1:2^bits
    %index=find(output==output(m));
    pixelspread_newglwise(graylvl_dist(i)+1)=pixelspread_newglwise(graylvl_dist(i)+1)+output(i);
end
figure,bar(pixelspread_newglwise,'m');
title('pixel distribution as per new gray levels');
xlabel('graylevels');ylabel('pixels');

%Unwrapping image basis new gray levels.
for i=1:dim(1)
    for j=1:dim(2)
        equalized_img(i,j)=graylvl_dist(a(i,j)+1);
    end
end
subplot(1,2,2),imshow(uint8(equalized_img));title('equalized image');
subplot(1,2,1),imshow(imread(image));title('original image');

        
        





