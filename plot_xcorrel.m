function plot_xcorrel(e,u)
% --- plot correlation based tests
% e is the vector of residuals
% u is the vector of inputs

maxlag = 100;

N = length(u);

% --- plots
conf_factor = 1.96/sqrt(N);
lag_vec = -maxlag:maxlag;
conf = [ones(length(lag_vec),1).*conf_factor ones(length(lag_vec),1).*-conf_factor];

figure
subplot(5,1,1)
EE = crossco(e,e,maxlag);
plot(lag_vec,EE,'k',lag_vec,conf,'k:')
xlim([-maxlag maxlag]);
ylim([-1 1]);
ylabel('$\phi_{\xi\xi}(\tau)$','Interpreter','LaTex')

subplot(5,1,2)
UE = crossco(u,e,maxlag);
plot(lag_vec,UE,'k',lag_vec,conf,'k:')
xlim([-maxlag maxlag]);
ylim([-1 1]);
ylabel('$\phi_{u\xi}(\tau)$','Interpreter','LaTex')

subplot(5,1,3)
EEU = crossco(e(1:end-1,1),e(2:end,1).*u(2:end,1),maxlag);
plot(lag_vec,EEU,'k',lag_vec,conf,'k:')
xlim([0 maxlag]);
ylim([-1 1]);
ylabel('$\phi_{\xi(\xi u)}(\tau)$','Interpreter','LaTex')

subplot(5,1,4)
U2E = crossco(u.^2 - mean(u.^2),e,maxlag);
plot(lag_vec,U2E,'k',lag_vec,conf,'k:')
xlim([-maxlag maxlag]);
ylim([-1 1]);
ylabel('$\phi_{(u^2)'' \xi}(\tau)$','Interpreter','LaTex')

subplot(5,1,5)
U2E2 = crossco(u.^2 - mean(u.^2),e.^2,maxlag);
plot(lag_vec,U2E2,'k',lag_vec,conf,'k:')
xlim([-maxlag maxlag]);
ylim([-1 1]);
ylabel('$\phi_{(u^2)'' \xi^2}(\tau)$','Interpreter','LaTex')
xlabel('$\tau$','Interpreter','LaTex')

end
