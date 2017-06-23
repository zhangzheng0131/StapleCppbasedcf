function results=run_StapleCpp(seq, res_path, bSaveImage)

close all;

x=seq.init_rect(1)-1;%matlab to c
y=seq.init_rect(2)-1;
w=seq.init_rect(3);
h=seq.init_rect(4);

%prepare img list
seq.path=[seq.path 'img/'];

command = ['python matlabImgLst.py -d ' seq.path ' -r "' num2str(x) ' ' num2str(y) ' ' num2str(w) ' ' num2str(h) ...
   '" -f imglst.lst' ' -s ' num2str(seq.startFrame) ' -e ' num2str(seq.endFrame) ' -n ' num2str(seq.nz) ' -x ' seq.ext];
dos(command);

tic
command = ['./tracker-benchmark imglst.lst'];
dos(command);
duration=toc;

results.res = dlmread('tracking_results.tmp');
results.res(:,1:2) =results.res(:,1:2) + 1;%c to matlab

results.type='rect';
results.fps=seq.len/duration;

results.fps = dlmread(['fps.tmp']);
