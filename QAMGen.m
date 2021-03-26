clear all
close all
clc

Q=4;
N_train=20000;
N_test=1000000;

Y_train=randi(2^Q,1,N_train)';
Y_test=randi(2^Q,1,N_test)';

Xt_train=dec2bin(Y_train-1);
Xr_train=2*bin2dec(Xt_train(:,1:Q/2))-2^(Q/2)+1;
Xi_train=2*bin2dec(Xt_train(:,Q/2+1:end))-2^(Q/2)+1;
X_train=Xr_train+j*Xi_train;

Xt_test=dec2bin(Y_test-1);
Xr_test=2*bin2dec(Xt_test(:,1:Q/2))-2^(Q/2)+1;
Xi_test=2*bin2dec(Xt_test(:,Q/2+1:end))-2^(Q/2)+1;
X_test=Xr_test+j*Xi_test;

TRAIN=[X_train,Y_train];
TEST=[X_test,Y_test];

save('TRAIN.mat', 'TRAIN')
save('TEST.mat', 'TEST')

Es=(X_train'*X_train)/N_train
Eb=Es/Q
plot(X_train,'*')

