# https://ardupilot.org/mavproxy/docs/getting_started/forwarding.html
cd ardupilot/ArduPlane
../Tools/autotest/sim_vehicle.py --map --console --out 127.0.0.1:14551 --out 127.0.0.1:14552 -streamrate 50

