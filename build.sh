set -e
#gcc -Wall -g -pg -pthread -std=c99 -c -O0 main.c
#gcc -Wall -g -pg -pthread -o main main.o -lm
gcc -Wall -pthread -std=c99 -c -O2 main.c
gcc -pthread -o main main.o -lm
#./main
