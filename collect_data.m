listing = dir('mengyue/*.mat');
% mac_spd = zeros(5864,16,900);
mac = 1;
bus = 52+mac;
line = [5];
for i=1:length(listing)
    filename = listing(i).name;
    tmp = load(['mengyue/' filename]);
%     mac_spd(i, :, :) = tmp.mac_spd(:, 101:1000);
%     fault1_cleared = find(abs(tmp.cur(87,:))<=1e-10,1);
%     fault2_cleared = find(abs(tmp.cur(88,:))<=1e-10,1);
%     fault_cleared = max([fault1_cleared, fault2_cleared]);
    fault_cleared = 1;
    
    data(i).bus_v = tmp.bus_v(bus, fault_cleared:end);
    data(i).cur = -tmp.cur(line, fault_cleared:end);
%     data(i).bus_freq = tmp.bus_freq(mac, fault_cleared:end);
    
    data(i).mac_ang = tmp.mac_ang(mac, fault_cleared:end);
    data(i).mac_spd = tmp.mac_spd(mac, fault_cleared:end);
    data(i).pelect = tmp.pelect(mac, fault_cleared:end);
    data(i).pmech = tmp.pmech(mac, fault_cleared:end);
    data(i).qelect = tmp.qelect(mac, fault_cleared:end);
    data(i).length = length(tmp.bus_freq(mac, :)) - fault_cleared + 1;
    data(i).filename = filename;
    data(i).bus2 = tmp.bus_v(2, fault_cleared:end);
end
save data_mac1_full_wBus2.mat data