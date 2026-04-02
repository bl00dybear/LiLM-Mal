#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <unistd.h>

#define SHM_SIZE 1024
#define SHM_KEY 0x1337

int main() {
    int shmid;
    char *shm_ptr;
    char secret_buffer[SHM_SIZE];

    shmid = shmget(SHM_KEY, SHM_SIZE, IPC_CREAT | 0666);
    if (shmid < 0) return 1;

    shm_ptr = shmat(shmid, NULL, 0);
    if (shm_ptr == (char *)-1) return 1;

    // Simulate capturing sensitive data
    FILE *f = fopen("/proc/self/comm", "r");
    if (f) {
        fgets(secret_buffer, sizeof(secret_buffer), f);
        fclose(f);
    }

    // Copying "captured" data to shared memory instead of disk/network
    memcpy(shm_ptr, secret_buffer, strlen(secret_buffer));

    detach:
        shmdt(shm_ptr);

    return 0;
}