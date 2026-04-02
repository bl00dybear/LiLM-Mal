### LiLM-Mal: Adversarial Testing Suite Documentation

| File | Type (Ground Truth) | Technical Logic | Testing Objective |
| :--- | :--- | :--- | :--- |
| **adversarial_ben_1.c** | **Benign** | Implements basic XOR encoding on a diagnostic string and performs a standard file size check on `/etc/hosts`. | **Sanity Check**: Evaluates if the model triggers on simple encoding patterns or basic system file access. |
| **adversarial_ben_2.c** | **Benign** | Mimics high-level malware behavior: uses `ptrace` for evasion, `fork` for daemonization, `mmap` for execution, and performs DNS/Socket beaconing. | **Shortcut Learning**: Tests if the model performs "Keyword Bingo" (flagging based on API names) or understands the harmless context. |
| **adversarial_ben_3.c** | **Benign** | (Innocent Overloader) An administrative health monitor. Attaches to PIDs via `ptrace` to read RIP registers, parses `/proc/net/tcp`, and sends UDP logs. | **False Positive Calibration**: Measures sensitivity toward aggressive system auditing tools that utilize "scary" syscalls for legitimate telemetry. |
| **adversarial_mal_1.c** | **Malware** | Userland Rootkit. Hijacks `open` and `readdir` via `LD_PRELOAD` interposition. Triggers a silent DNS beacon upon access to `/etc/shadow`. | **Recall & Intent Detection**: Verifies if the model can identify library hooking, stealthy exfiltration, and malicious persistence techniques. |
| **adversarial_mal_2.c** | **Malware** | Executes extensive "junk" mathematical operations (bubble sorts, square roots) to delay execution. Buries a reverse shell (`socket`, `dup2`, `execve`) at the very end. | **Attention Resiliency**: Tests if the MIL mechanism can isolate the "needle in the haystack" (the payload) despite the overwhelming volume of benign noise. |
| **adversarial_mal_3.c** | **Malware** | Uses a Caesar cipher masked as an `i18n` (internationalization) function to decode and execute a system command. | **Semantic Analysis**: Tests if the model is misled by benign function names or if it tracks data flow into critical sinks like `system()`. |
| **adversarial_mal_4.c** | **Malware** | Utilizes Shared Memory IPC (`shmget`, `shmat`) to exfiltrate system information to another local process. | **Exotic Syscall Coverage**: Evaluates the model's ability to recognize Inter-Process Communication as a valid exfiltration or spying vector. |

