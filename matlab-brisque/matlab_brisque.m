jpgFiles = dir('../../../sample_imgs/*.jpg');
pngFiles = dir('../../../sample_imgs/*.png');
dirlist = [jpgFiles; pngFiles];
fileID = fopen('output.txt', 'w');

for i = 1:length(dirlist)
    fullPath = fullfile(dirlist(i).folder, dirlist(i).name);
    I = imread(fullPath);
    brisqueI = brisque(I);
    fprintf(fileID, "BRISQUE score for " + dirlist(i).name + " image is: " + brisqueI + "\n");
end