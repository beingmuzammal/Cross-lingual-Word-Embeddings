//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

typedef double real;

typedef struct cooccur_rec {
    int lan1;
    int lan2;
    int word1;
    int word2;
    real val;
} CREC;

typedef long long bigint;

CREC rec;

int i = 1;
bigint count = 0;
bigint last_count = 0;

#define MAX_SIZE 100000
real countx[MAX_SIZE];
real county[MAX_SIZE];
real allcount = 0.0;
real tmpreal = 0.0;

char *save_file, *save_file_d, *save_file_i, *save_file_p, *save_file_pmi, *input_file;


static const int MAX_STRING_LENGTH = 1000;
/* Efficient string comparison */
int scmp( char *s1, char *s2 ) {
    while(*s1 != '\0' && *s1 == *s2) {s1++; s2++;}
    return(*s1 - *s2);
}

int find_arg(char *str, int argc, char **argv) {
    int i;
    for (i = 1; i < argc; i++) {
        if(!scmp(str, argv[i])) {
            if (i == argc - 1) {
                printf("No argument given for %s\n", str);
                exit(1);
            }
            return i;
        }
    }
    return -1;
}

real fadd(real *start, real *end) {
  if (start == end)
    return 0.0;

  if (end == start + 1)
    return *start;

  real *middle = start + (end - start) / 2;

  return fadd(start, middle) + fadd(middle, end);
}

int summarize() {
  FILE *fd, *fi, *fp, *fpmi, *finput;
  finput = fopen(input_file, "rb");
  while (1) {
    fread(&rec, sizeof(CREC), 1, finput);
    if(feof(finput)) {
      break;
    }
    countx[rec.word1] += rec.val;
    county[rec.word2] += rec.val;
  }
  fclose(finput);

  allcount = fadd(countx, countx + MAX_SIZE) + fadd(county, county + MAX_SIZE);
  allcount *= 0.5;

  fd = fopen(save_file_d, "w");
  fi = fopen(save_file_i, "w");
  fp = fopen(save_file_p, "w");
  fpmi = fopen(save_file_pmi, "w");
  finput = fopen(input_file, "rb");
  i = 1;
  while (1) {
    fread(&rec, sizeof(CREC), 1, finput);
    while (rec.word1 > i) {
      //fwrite(&i, sizeof(int), 1, stdout);
      fwrite(&count, sizeof(bigint), 1, fp);
      last_count = count;
      //fwrite(&countx, sizeof(real), 1, stdout);
      //fwrite(&county, sizeof(real), 1, stdout);
      ++i;
      //countx = 0.0;
      //county = 0.0;
    }

    if(feof(finput)) {
      //fwrite(&i, sizeof(int), 1, stdout);
      fwrite(&count, sizeof(bigint), 1, fp);
      last_count = count;
      //fwrite(&countx, sizeof(real), 1, stdout);
      //fwrite(&county, sizeof(real), 1, stdout);
      break;
    }

    count += 1;
    fwrite(&rec.val, sizeof(real), 1, fd);
    tmpreal = allcount / countx[rec.word1] / county[rec.word2] * rec.val;
    fwrite(&tmpreal, sizeof(real), 1, fpmi);
    fwrite(&rec.word2, sizeof(int), 1, fi);
  }

  fclose(fd);
  fclose(fi);
  fclose(fp);
  fclose(fpmi);
  fclose(finput);

  return 1;
}

int main(int argc, char **argv) {
    save_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
    input_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
    save_file_d = malloc(sizeof(char) * MAX_STRING_LENGTH);
    save_file_i = malloc(sizeof(char) * MAX_STRING_LENGTH);
    save_file_p = malloc(sizeof(char) * MAX_STRING_LENGTH);
    save_file_pmi = malloc(sizeof(char) * MAX_STRING_LENGTH);

    if ((i = find_arg((char *)"-save-file", argc, argv)) > 0) strcpy(save_file, argv[i + 1]);
    else strcpy(save_file, (char *)"cooccur");

    if ((i = find_arg((char *)"-input-file", argc, argv)) > 0) strcpy(input_file, argv[i + 1]);
    else strcpy(input_file, (char *)"cooccur");


    sprintf(save_file_d,"%s.d",save_file);
    sprintf(save_file_i,"%s.i",save_file);
    sprintf(save_file_p,"%s.p",save_file);
    sprintf(save_file_pmi,"%s.pmi",save_file);

    for (i = 0; i < MAX_SIZE; ++i) {
      countx[i] = 0;
      county[i] = 0;
    }
    return summarize();
}