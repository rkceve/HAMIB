# Pod procedure (what Ryosuke does, what Fable does)

Ryosuke (once):
1. RunPod → Pods → template "RunPod PyTorch 2.x / CUDA 12.x" → GPU per the environment choice
   (β: A100 80 GB PCIe or H100 80 GB; α: RTX A6000 48 GB) → volume ≥ 150 GB → expose TCP port 22.
2. Console → Settings → "SSH Public Keys": paste the public key whose PRIVATE key stays on this PC.
3. Give Fable: the pod's public IP, the SSH port, and the private-key PATH as an environment variable
   name (`MCB_POD_KEY`). Never paste key contents into the chat.
4. Register a payment method on TypeSafe or keep the Jev budget guard at the current balance.

Fable (over `ssh -i "$MCB_POD_KEY" -p <port> root@<ip>`):
1. `MCB_REPO_URL=<clone url> bash bootstrap_beta.sh mcbuild-bench` (variant β) — installs, downloads
   weights, runs the CPU self-tests on the pod. Nothing is generated yet.
2. V2: load the reader, assert sdpa / 16 full-attention layers / fla kernels / memory after one forward.
3. V3: injection probe on a 300-token block (counters exact, attention share rises with w) → w grid
   frozen by Ryosuke.
4. V4: `build_cd` on the first 3 round trips with `--max-jev-input-tokens` inside the balance →
   Ryosuke inspects the tree; Jev spend extrapolated to 36 round trips.
5. V5: baseline on 5 questions with chunked prefill (memory, seconds/question).
6. Main run, then scoring locally.

Destructive pod operations (stop / terminate / delete volume) only with Ryosuke's approval each time.
