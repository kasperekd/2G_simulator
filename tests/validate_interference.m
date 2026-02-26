clear; clc; close all;

load('debug_interference_check.mat');

sig = double(signal_ant1(:));
int = double(interf_ant1(:));

% 2. во временной области
figure('Name', 'Time Domain');
subplot(2,1,1);
plot(abs(sig)); hold on; plot(abs(int), 'r');
title('Amplitude Envelope');
legend('Useful Signal', 'Interference');
grid on;

subplot(2,1,2);
plot(real(int)); hold on; plot(imag(int), 'r--');
title('Interference I/Q components');
legend('I', 'Q');
grid on;

% 3. PSD
figure('Name', 'Power Spectral Density');
pwelch(sig, [], [], [], fs, 'centered');
hold on;
pwelch(int, [], [], [], fs, 'centered');
legend('Signal', 'Interference');
title('Comparison of Spectra');

% 4. PAPR / CCDF
figure('Name', 'CCDF');
ccdf_sig = comm.CCDF('AveragePowerOutputPort', true);
ccdf_int = comm.CCDF('AveragePowerOutputPort', true);

[p_sig, ~, papr_sig] = ccdf_sig(sig);
[p_int, ~, papr_int] = ccdf_int(int);

plot(ccdf_sig);
hold on;
plot(ccdf_int);
legend(['Signal (PAPR=' num2str(papr_sig) ' dB)'], ...
       ['Interference (PAPR=' num2str(papr_int) ' dB)']);
title('CCDF Comparison');