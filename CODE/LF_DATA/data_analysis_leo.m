%% Explore MATLAB dataset

clear;
close all;
clc;

load data.mat

%% xn - nominal states

figure('Name','xn');

subplot(2,1,1)
plot(squeeze(xn(1,:,:)))
title('xn - Position')
xlabel('Time')
ylabel('Position')
grid on

subplot(2,1,2)
plot(squeeze(xn(2,:,:)))
title('xn - Velocity')
xlabel('Time')
ylabel('Velocity')
grid on


%% yn - nominal observations

figure('Name','yn');

plot(squeeze(yn(1,:,:)))
title('yn - Measurements')
xlabel('Time')
ylabel('Measurement')
grid on


%% xw - least favorable states

figure('Name','xw');

subplot(2,1,1)
hold on
for j = 1:size(xw,4)
    plot(squeeze(xw(1,:,:,j)))
end
title('xw - Position')
xlabel('Time')
ylabel('Position')
grid on

subplot(2,1,2)
hold on
for j = 1:size(xw,4)
    plot(squeeze(xw(2,:,:,j)))
end
title('xw - Velocity')
xlabel('Time')
ylabel('Velocity')
grid on


%% yw - least favorable observations

figure('Name','yw');

hold on
for j = 1:size(yw,4)
    plot(squeeze(yw(1,:,:,j)))
end
title('yw - Measurements')
xlabel('Time')
ylabel('Measurement')
grid on


%% Robust covariance VR

figure('Name','VR');

subplot(2,2,1)
plot(squeeze(VR(1,1,:)),'LineWidth',2)
title('VR(1,1)')
grid on

subplot(2,2,2)
plot(squeeze(VR(1,2,:)),'LineWidth',2)
title('VR(1,2)')
grid on

subplot(2,2,3)
plot(squeeze(VR(2,1,:)),'LineWidth',2)
title('VR(2,1)')
grid on

subplot(2,2,4)
plot(squeeze(VR(2,2,:)),'LineWidth',2)
title('VR(2,2)')
grid on


%% Nominal covariance PR

figure('Name','PR');

subplot(2,2,1)
plot(squeeze(PR(1,1,2:end)),'LineWidth',2)
title('PR(1,1)')
grid on

subplot(2,2,2)
plot(squeeze(PR(1,2,2:end)),'LineWidth',2)
title('PR(1,2)')
grid on

subplot(2,2,3)
plot(squeeze(PR(2,1,2:end)),'LineWidth',2)
title('PR(2,1)')
grid on

subplot(2,2,4)
plot(squeeze(PR(2,2,2:end)),'LineWidth',2)
title('PR(2,2)')
grid on


%% Comparison PR vs VR

figure('Name','Variance comparison')

subplot(2,1,1)
plot(squeeze(PR(1,1,2:end)),'LineWidth',2)
hold on
plot(squeeze(VR(1,1,2:end)),'LineWidth',2)
legend('PR','VR')
title('Position variance')
grid on

subplot(2,1,2)
plot(squeeze(PR(2,2,2:end)),'LineWidth',2)
hold on
plot(squeeze(VR(2,2,2:end)),'LineWidth',2)
legend('PR','VR')
title('Velocity variance')
grid on

%% ========================================================================
%% Mean ± Std summaries
%% ========================================================================

%% xn

t_xn = 0:size(xn,2)-1;

pos_mean = mean(squeeze(xn(1,:,:)),2);
pos_std  = std(squeeze(xn(1,:,:)),0,2);

vel_mean = mean(squeeze(xn(2,:,:)),2);
vel_std  = std(squeeze(xn(2,:,:)),0,2);

figure('Name','xn statistics')

subplot(2,1,1)
hold on
fill([t_xn fliplr(t_xn)], ...
     [(pos_mean-pos_std)' fliplr((pos_mean+pos_std)')], ...
     [0.85 0.85 1], ...
     'EdgeColor','none','FaceAlpha',0.5)
plot(t_xn,pos_mean,'b','LineWidth',2)
title('xn Position - Mean \pm Std')
xlabel('Time')
ylabel('Position')
grid on

subplot(2,1,2)
hold on
fill([t_xn fliplr(t_xn)], ...
     [(vel_mean-vel_std)' fliplr((vel_mean+vel_std)')], ...
     [0.85 0.85 1], ...
     'EdgeColor','none','FaceAlpha',0.5)
plot(t_xn,vel_mean,'b','LineWidth',2)
title('xn Velocity - Mean \pm Std')
xlabel('Time')
ylabel('Velocity')
grid on


%% yn

t_yn = 1:size(yn,2);

yn_mean = mean(squeeze(yn(1,:,:)),2);
yn_std  = std(squeeze(yn(1,:,:)),0,2);

figure('Name','yn statistics')

hold on
fill([t_yn fliplr(t_yn)], ...
     [(yn_mean-yn_std)' fliplr((yn_mean+yn_std)')], ...
     [0.85 1 0.85], ...
     'EdgeColor','none','FaceAlpha',0.5)
plot(t_yn,yn_mean,'g','LineWidth',2)
title('yn Measurements - Mean \pm Std')
xlabel('Time')
ylabel('Measurement')
grid on


%% xw

t_xw = 0:size(xw,2)-1;

xw_pos = reshape(xw(1,:,:,:),size(xw,2),[]);
xw_vel = reshape(xw(2,:,:,:),size(xw,2),[]);

xw_pos_mean = mean(xw_pos,2);
xw_pos_std  = std(xw_pos,0,2);

xw_vel_mean = mean(xw_vel,2);
xw_vel_std  = std(xw_vel,0,2);

figure('Name','xw statistics')

subplot(2,1,1)
hold on
fill([t_xw fliplr(t_xw)], ...
     [(xw_pos_mean-xw_pos_std)' fliplr((xw_pos_mean+xw_pos_std)')], ...
     [1 0.9 0.8], ...
     'EdgeColor','none','FaceAlpha',0.5)
plot(t_xw,xw_pos_mean,'r','LineWidth',2)
title('xw Position - Mean \pm Std')
xlabel('Time')
ylabel('Position')
grid on

subplot(2,1,2)
hold on
fill([t_xw fliplr(t_xw)], ...
     [(xw_vel_mean-xw_vel_std)' fliplr((xw_vel_mean+xw_vel_std)')], ...
     [1 0.9 0.8], ...
     'EdgeColor','none','FaceAlpha',0.5)
plot(t_xw,xw_vel_mean,'r','LineWidth',2)
title('xw Velocity - Mean \pm Std')
xlabel('Time')
ylabel('Velocity')
grid on


%% yw

t_yw = 1:size(yw,2);

yw_all = reshape(yw(1,:,:,:),size(yw,2),[]);

yw_mean = mean(yw_all,2);
yw_std  = std(yw_all,0,2);

figure('Name','yw statistics')

hold on
fill([t_yw fliplr(t_yw)], ...
     [(yw_mean-yw_std)' fliplr((yw_mean+yw_std)')], ...
     [1 0.9 0.8], ...
     'EdgeColor','none','FaceAlpha',0.5)
plot(t_yw,yw_mean,'r','LineWidth',2)
title('yw Measurements - Mean \pm Std')
xlabel('Time')
ylabel('Measurement')
grid on


%% Trace comparison

tracePR = squeeze(PR(1,1,:) + PR(2,2,:));
traceVR = squeeze(VR(1,1,:) + VR(2,2,:));

figure('Name','Trace(PR) vs Trace(VR)')

plot(tracePR(2:end),'LineWidth',2)
hold on
plot(traceVR(2:end),'LineWidth',2)

legend('trace(PR)','trace(VR)','Location','best')
xlabel('Time')
ylabel('Trace')
title('Total covariance')
grid on