#ifndef __LOG_H__
#define __LOG_H__

#ifdef __cplusplus
extern "C" {
#endif
    
char *strstrip(char *s);

int   getpos_t();
char* getarg_t();
int   getopt_t(int argc, char *argv[], char *opts);

double timeStamp();
int    isExpired(int year, int month);

#ifdef __cplusplus
}
#endif

#endif
