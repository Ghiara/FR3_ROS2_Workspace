# Connect Franka FR3 with PC

## Step 1: configuration in Franka Robot

1. direct connect franka robot from its left side cable interface (not control box)

2. login franka desk: `https://robot.franka.de/desk/`

```yaml

account: franka

PIN: franka123

```

3. under `settings` tab (above left), `Network` section, unbox DHCP client, manually setup following configs:

```yaml

Address 192.168.1.2

Netmask 255.255.255.0

Gateway 255.255.255.254

DNS 8.8.8.8

```

4. reboot the franka system


## Step 2: configuration in control box

1. connect the cable with control box

2. in PC `USB Ethernet` connection option:

    - 2.1 under IPv4 tab, activate `manual` mode
    - 2.2 for addresses section, setup following:
    ```yaml

        Address 192.168.1.3

        Gateway 255.255.255.0

        DNS 8.8.8.8

    ```
    - 2.3 save the config

3. Open a terminal, try if connect with control box:

```bash

ping 192.168.1.2

```

4. In the brower (e.g., Chrome), use IP `192.168.1.2` to login the franka desk.