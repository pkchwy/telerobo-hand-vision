import os
import pty
import serial
import time
import subprocess

# Config
VIRTUAL_PORT_LINK = '/tmp/robot_write'

def simulate_robot():
    # Refuse to start if another instance is already alive — otherwise its
    # finally-block on shutdown would yank the symlink out from under us
    # and leave the surviving instance with no /tmp/robot_write.
    if os.path.lexists(VIRTUAL_PORT_LINK):
        target = os.readlink(VIRTUAL_PORT_LINK) if os.path.islink(VIRTUAL_PORT_LINK) else None
        if target and os.path.exists(target):
            raise SystemExit(
                f"[fake_robot] {VIRTUAL_PORT_LINK} already points at a live pty "
                f"({target}). Another fake_robot.py is running — kill it first."
            )
        os.remove(VIRTUAL_PORT_LINK)  # stale symlink from a crashed run

    master, slave = pty.openpty()
    slave_name = os.ttyname(slave)
    os.symlink(slave_name, VIRTUAL_PORT_LINK)
    
    print(f"[*] Fake Robot Started")
    print(f"[*] Virtual Port: {slave_name} -> {VIRTUAL_PORT_LINK}")
    print(f"[*] Simulated Baud Rate: 9600")
    print("-" * 50)
    print("Robot hand ready - waiting for servo commands")

    buffer = ""
    try:
        while True:
            # Read from master (what server.py writes to the slave)
            data = os.read(master, 1024).decode('utf-8')
            if not data:
                continue
            
            buffer += data
            if '\n' in buffer:
                lines = buffer.split('\n')
                for command in lines[:-1]:
                    command = command.strip()
                    if not command:
                        continue
                    
                    # Simulate parseServoCommand logic from lehand.ino
                    try:
                        parts = command.split(',')
                        if len(parts) == 5:
                            # Constrain values 500-2500
                            values = [max(500, min(2500, int(v))) for v in parts]
                            
                            # Print visual feedback
                            print(f"[ROBOT] Received: {values}")
                            
                            # Send back confirmation string like Arduino
                            response = f"Servos updated: {','.join(map(str, values))}\n"
                            os.write(master, response.encode('utf-8'))
                        else:
                            print(f"[ERROR] Invalid command format: {command}")
                    except ValueError:
                        print(f"[ERROR] Non-numeric data: {command}")
                
                buffer = lines[-1]
                
    except KeyboardInterrupt:
        print("\n[*] Stopping Fake Robot...")
    finally:
        if os.path.lexists(VIRTUAL_PORT_LINK):
            os.remove(VIRTUAL_PORT_LINK)

if __name__ == "__main__":
    simulate_robot()
