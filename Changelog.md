
# Change Log
All notable changes to this project will be documented in this file.

## 2020-08-06
Dedicated LSTM network
Settings: Full trajectories, interpolation x5, length 700 time steps, random start.
LSTM Networks: 128 -> 128(Va, mac_ang); 192 -> 192(others)

## 2020-08-03
Dedicated/Shared LSTM network(not fully dedicated, loss function is shared).

## 2020-07-27
Switched to post-clearance data

## 2020-07-16
Modularization

Added in interpolation: x5

## 2020-07-12
Changelog created.

Former features:

Added in bus 2 voltage, resulted in better performance

Removed bus_freq and a5 faults(which is bad for mac1)

Change teacher forcing between iterations: faster training
