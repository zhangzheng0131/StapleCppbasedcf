function results=run_StapleCpp(seq, res_path, bSaveImage)

close all;

x=seq.init_rect(1)-1;%matlab to c
y=seq.init_rect(2)-1;
w=seq.init_rect(3);
h=seq.init_rect(4);

%prepare img list

command = ['python matlabImgLst.py -d ' seq.path ' -r "' num2str(x) ' ' num2str(y) ' ' num2str(w) ' ' num2str(h) ...
   '" -f imglst.lst' ];
dos(command);

tic
command = ['demo_img_list.exe imglst.lst'];
dos(command);
duration=toc;

results.res = dlmread('tracking_results.txt');
results.res(:,1:2) =results.res(:,1:2) + 1;%c to matlab

results.type='rect';
results.fps=seq.len/duration;

%results.fps = dlmread([seq.name '_ST_FPS.txt']);
