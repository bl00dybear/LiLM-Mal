#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/ptrace.h>
#include <sys/user.h>
#include <sys/wait.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>

typedef struct {
    int pid;
    unsigned long last_rip;
    char status[16];
} proc_stats_t;

void report_stats(proc_stats_t *s) {
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) return;

    struct sockaddr_in serv;
    serv.sin_family = AF_INET;
    serv.sin_port = htons(8888);
    serv.sin_addr.s_addr = inet_addr("192.168.1.100");

    char buffer[256];
    snprintf(buffer, sizeof(buffer), "STAT|PID:%d|RIP:%lx|ST:%s", s->pid, s->last_rip, s->status);
    
    sendto(sock, buffer, strlen(buffer), 0, (struct sockaddr*)&serv, sizeof(serv));
    close(sock);
}

void audit_network() {
    FILE *fp = fopen("/proc/net/tcp", "r");
    if (!fp) return;

    char line[256];
    int count = 0;
    while (fgets(line, sizeof(line), fp)) {
        count++;
    }
    fclose(fp);
}

void trace_process(int pid) {
    struct user_regs_struct regs;
    proc_stats_t stats;

    if (ptrace(PTRACE_ATTACH, pid, NULL, NULL) < 0) return;
    waitpid(pid, NULL, 0);

    if (ptrace(PTRACE_GETREGS, pid, NULL, &regs) == 0) {
        stats.pid = pid;
        stats.last_rip = regs.rip;
        strncpy(stats.status, "ACTIVE", 16);
        report_stats(&stats);
    }

    ptrace(PTRACE_DETACH, pid, NULL, NULL);
}

int main(int argc, char **argv) {
    if (argc < 2) return 1;
    
    int target_pid = atoi(argv[1]);
    
    if (fork() == 0) {
        setsid();
        while (1) {
            audit_network();
            trace_process(target_pid);
            sleep(60);
        }
    }

    return 0;
}