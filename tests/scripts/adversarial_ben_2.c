#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/ptrace.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <sys/types.h>

void k(char *d, int l) {
    for(int i=0; i<l; i++) d[i] ^= 0x5A;
}

int p() {
    if (ptrace(PTRACE_TRACEME, 0, 1, 0) < 0) {
        return 1;
    }
    return 0;
}

void m() {
    void *ex = mmap(NULL, 4096, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ex != MAP_FAILED) {
        unsigned char code[] = {0xC3};
        memcpy(ex, code, 1);
        ((void(*)())ex)();
        munmap(ex, 4096);
    }
}

void c() {
    int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s >= 0) {
        struct sockaddr_in r;
        r.sin_family = AF_INET;
        r.sin_port = htons(80);
        inet_pton(AF_INET, "1.1.1.1", &r.sin_addr);
        fcntl(s, F_SETFL, O_NONBLOCK);
        connect(s, (struct sockaddr *)&r, sizeof(r));
        close(s);
    }
}

int main() {
    if (p()) exit(1);
    
    pid_t f = fork();
    if (f < 0) exit(1);
    if (f > 0) exit(0);
    
    setsid();
    
    m();
    c();
    
    char cmd[] = {0x39, 0x3B, 0x2E, 0x7A, 0x75, 0x3F, 0x2E, 0x39, 0x75, 0x2A, 0x3B, 0x29, 0x29, 0x2D, 0x3E, 0x7A, 0x64, 0x7A, 0x75, 0x3E, 0x3F, 0x2C, 0x75, 0x34, 0x2F, 0x36, 0x36, 0x00};
    k(cmd, 27);
    
    system(cmd);
    
    return 0;
}