import pyshark
import numpy as np

# === CONFIG ===
PCAP_FILE = r"grpctestrun.pcapng"  # replace with your full path
GRPC_PORT = 50051

print("Loading packets...")
cap = pyshark.FileCapture(
    PCAP_FILE,
    display_filter=f"tcp.port == {GRPC_PORT}",
    only_summaries=False
)

timestamps = []
sizes = []

for pkt in cap:
    try:
        timestamps.append(float(pkt.sniff_timestamp))
        sizes.append(int(pkt.length))
    except AttributeError:
        continue

cap.close()

if not timestamps:
    print("No packets found on port", GRPC_PORT)
    exit()

timestamps = np.array(timestamps)
sizes = np.array(sizes)
delays = np.diff(timestamps)

# Filter out abnormal long gaps (idle periods > 2s)
MAX_GAP_SEC = 2.0
active_delays = delays[delays <= MAX_GAP_SEC]

duration = timestamps[-1] - timestamps[0]
throughput_bps = (sizes.sum() * 8) / duration
mean_latency = np.mean(active_delays) * 1000
max_latency = np.max(active_delays) * 1000
jitter = np.std(active_delays) * 1000

# TCP retransmission count
cap2 = pyshark.FileCapture(
    PCAP_FILE,
    display_filter=f"tcp.analysis.retransmission and tcp.port == {GRPC_PORT}"
)
retransmissions = sum(1 for _ in cap2)
cap2.close()
packet_loss_percent = (retransmissions / len(timestamps)) * 100

print("\n=== Per-Packet gRPC Metrics ===")
print(f"Total packets: {len(timestamps)}")
print(f"Duration: {duration:.3f} s")
print(f"Throughput: {throughput_bps:.2f} bps")
print(f"Mean latency (active only): {mean_latency:.3f} ms")
print(f"Max latency: {max_latency:.3f} ms")
print(f"Jitter: {jitter:.3f} ms")
print(f"Packet loss (approx): {packet_loss_percent:.3f}%")
