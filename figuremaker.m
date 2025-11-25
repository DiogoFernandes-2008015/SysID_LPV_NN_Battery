load('resfig.mat')
figure(1)
subplot(2,2,1)
hold on
title('Estimated response of the best hybrid model - Training Dataset')
xlabel('Time (s)')
ylabel('Terminal Voltage (V)')
grid
plot(t,v)
plot(t2,ywnnt,'--r')
legend('Experimental Data','WNN - Hat')
hold off
subplot(2,2,3)
hold on
title('Residue of the hybrid models - Training Dataset')
xlabel('Time (s)')
ylabel('Terminal Voltage (mV)')
grid
plot(t2,1000*(v2-ywnnt),'-r')
plot(t2,1000*(v2-yrbftht),'-k')
plot(t2,1000*(v2-yrelut),'-b')
legend('WNN-Hat','RBF-TH','MLP-ReLu','Location','best')
hold off
subplot(2,2,2)
hold on
title('Detail of the estimation - Training Dataset')
xlabel('Time (s)')
ylabel('Terminal Voltage (V)')
grid
plot(t(15000:16000), v(15000:16000))
plot(t2(15000/2:16000/2), ywnnt(15000/2:16000/2), '--r')
plot(t2(15000/2:16000/2), yrelut(15000/2:16000/2), '--k')
plot(t2(15000/2:16000/2), yrelut(15000/2:16000/2), '--b')
axis([1.5e4 1.6e4 3.5 3.85])
legend('Experimental Data','WNN - Hat','RBF-TH','MLP-ReLu','Location','best')
hold off
subplot(2,2,4)
hold on
title('Detail of the Residue - Training Dataset')
xlabel('Time (s)')
ylabel('Terminal Voltage (mV)')
grid
plot(t2(15000/2:16000/2),1000*(v2(15000/2:16000/2)-ywnnt(15000/2:16000/2)),'-r')
plot(t2(15000/2:16000/2),1000*(v2(15000/2:16000/2)-yrbftht(15000/2:16000/2)),'-k')
plot(t2(15000/2:16000/2),1000*(v2(15000/2:16000/2)-yrelut(15000/2:16000/2)),'-b')
legend('WNN-Hat','RBF-TH','MLP-ReLu','Location','best')
hold off
load('data_val.mat')
figure(2)
subplot(2,2,1)
hold on
title('Estimated response of the best hybrid model - Validation Dataset')
xlabel('Time (s)')
ylabel('Terminal Voltage (V)')
grid
plot(t,v)
plot(t,ywnnv,'--r')
legend('Experimental Data','WNN - Hat')
hold off
subplot(2,2,3)
hold on
title('Residue of the hybrid models - Validation Dataset')
xlabel('Time (s)')
ylabel('Terminal Voltage (mV)')
grid
plot(t,abs(1000*(v-ywnnv)),'-r')
plot(t,abs(1000*(v-yrbfthv)),'-k')
plot(t,abs(1000*(v-yreluv)),'-b')
legend('WNN-Hat','RBF-TH','MLP-ReLu','Location','best')
hold off
subplot(2,2,2)
hold on
title('Detail of the estimation - Validation Dataset')
xlabel('Time (s)')
ylabel('Terminal Voltage (V)')
grid
plot(t(7000:8000), v(7000:8000))
plot(t(7000:8000), ywnnv(7000:8000), '--r')
plot(t(7000:8000), yreluv(7000:8000), '--k')
plot(t(7000:8000), yreluv(7000:8000), '--b')
axis([7e3 8e3 3.5 4])
legend('Experimental Data','WNN - Hat','RBF-TH','MLP-ReLu','Location','best')
hold off
subplot(2,2,4)
hold on
title('Detail of the Residue - Training Dataset')
xlabel('Time (s)')
ylabel('Terminal Voltage (mV)')
grid
plot(t(7000:8000),1000*(abs(v(7000:8000)-ywnnv(7000:8000))),'-r')
plot(t(7000:8000),1000*(abs(v(7000:8000)-yrbfthv(7000:8000))),'-k')
plot(t(7000:8000),1000*(abs(v(7000:8000)-yreluv(7000:8000))),'-b')
legend('WNN-Hat','RBF-TH','MLP-ReLu','Location','best')
hold off
load('data_train.mat')
figure(3)
subplot(1,2,1)
title('Scatter plot - Training Dataset')
hold on
minY = min([min(v), min(ywnnt)]);
maxY = max([max(v), max(ywnnt)]);
plot([minY, maxY], [minY, maxY], 'k-', 'LineWidth', 2, 'DisplayName', 'Perfect model');
xlabel('Real');
ylabel('Prediction');
scatter(v(1:2:end), ywnnt, 'blue', 'DisplayName', 'Prediction');
grid
hold off
load('data_val.mat')
subplot(1,2,2)
title('Scatter plot - Validation Dataset')
hold on
minY = min([min(v), min(ywnnv)]);
maxY = max([max(v), max(ywnnv)]);
plot([minY, maxY], [minY, maxY], 'k-', 'LineWidth', 2, 'DisplayName', 'Perfect model');
xlabel('Real');
ylabel('Prediction');
scatter(v, ywnnv, 'blue', 'DisplayName', 'Prediction');
grid
hold off

load('data_train.mat')
plot_xcorrel(v(1:2:end)-ywnnt,i(1:2:end))
subplot(5,1,1)
title('Correlation test - Training Dataset')


load('data_val.mat')
plot_xcorrel(v-ywnnv,i)
subplot(5,1,1)
title('Correlation test - Validation Dataset')
