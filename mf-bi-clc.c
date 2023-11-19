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
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define _FILE_OFFSET_BITS 64
#define MAX_STRING_LENGTH 1000
#define INFINITE_SMALL 1e-6

typedef double real;

typedef struct cooccur_rec {
    int lan1;
    int lan2;
    int word1;
    int word2;
    real val;
} CREC;

int verbose = 2; // 0, 1, or 2
int num_threads = 8; // pthreads
int num_iter = 25; // Number of full passes through cooccurrence matrix
int vector_size = 50; // Word vector size
int use_binary = 1; // 0: save as text files; 1: save as binary; 2: both. For binary, save both word and context word vectors.
int dump_all = 0;
int model = 2; // For text file output only. 0: concatenate word and context vectors (and biases) i.e. save everything; 1: Just save word vectors (no bias); 2: Save (word + context word) vectors (no biases)
real eta = 0.05; // Initial learning rate
real init_eta = 0.05;
real alpha = 0.75, x_max1 = 100.0, x_max2, x_max_bi; // Weighting function parameters, not extremely sensitive to corpus, though may need adjustment for very small or very large corpora
real *W1, *W2, *cost;
real matrixbias1 = 1.0, matrixbias2 = 1.0, matrixbias_bi = 1.0;
long long num_lines, *lines_per_thread, vocab_size1, vocab_size2, total_size;
char *vocab_file1, *vocab_file2, *save_W1_file, *save_W2_file;
char *input_file1, *input_file2, *input_file_bi;
char *input_file_d1, *input_file_i1, *input_file_p1, *input_file_pmi1;
char *input_file_d2, *input_file_i2, *input_file_p2, *input_file_pmi2;
char *input_file_d_bi, *input_file_i_bi, *input_file_p_bi, *input_file_pmi_bi;
real weight1 = 1.0, weight2 = 1.0, weightbi = 1.0;
real sample1 = 1.0, sample2 = 1.0, samplebi = 1.0;

real negative_ratio = 0.0;
real negative_ratio_bi = 0.0;
real negative_val;
real negative_pmi = 1e-2;
real min_pmi = 0.0;

long long *deviation1;
long long *deviation2;
long long *deviation_bi;
real *bias_w, *bias_w2;
real *bias_c, *bias_c2;
real **positive;
real **pmi;
int **positions;
long long count;
real val;

int *thread_split;
int *thread_split_m;
int *start_per_thread;

real lambda = 0.001;
real init_lambda = 0.001;

int global_seed = 8913;


real *buffer_d1, *buffer_d2, *buffer_d_bi;
int *buffer_i1, *buffer_i2, *buffer_i_bi;
real *buffer_pmi1, *buffer_pmi2, *buffer_pmi_bi;

void swap_int(int *a, int *b) {
  int temp;
  temp = *a;
  *a = *b;
  *b = temp;
  return;
}

/* Efficient string comparison */
int scmp( char *s1, char *s2 ) {
    while(*s1 != '\0' && *s1 == *s2) {s1++; s2++;}
    return(*s1 - *s2);
}

real sign(real val) {
    if (val > INFINITE_SMALL)
        return 1.0;
    else if (val < -INFINITE_SMALL)
        return -1.0;
    else
        return 0;
}

static const long LRAND_MAX = ((long) RAND_MAX + 2) * (long)RAND_MAX;
/* Generate uniformly distributed random long ints */
long rand_long(long n) {
    long limit = LRAND_MAX - LRAND_MAX % n;
    long rnd;
    do {
        rnd = ((long)RAND_MAX + 1) * (long)rand_r(&global_seed) + (long)rand_r(&global_seed);
    } while (rnd >= limit);
    return rnd % n;
}

void initialize_parameters() {
    long long a, b;
    vector_size++; // Temporarily increment to allocate space for bias

    /* Allocate space for word vectors and context word vectors, and correspodning gradsq */
    a = posix_memalign((void **)&W1, 128, 2 * vocab_size1 * vector_size * sizeof(real)); // Might perform better than malloc
    if (W1 == NULL) {
        fprintf(stderr, "Error allocating memory for W\n");
        exit(1);
    }
    a = posix_memalign((void **)&W2, 128, 2 * vocab_size2 * vector_size * sizeof(real)); // Might perform better than malloc
    if (W2 == NULL) {
        fprintf(stderr, "Error allocating memory for W\n");
        exit(1);
    }
    bias_w = (real *)malloc(sizeof(real) * (vocab_size1 + 2));
    for (b = 0; b < vocab_size1; ++b) {
        bias_w[b + 1] = (rand() / (real)RAND_MAX - 0.5) / vector_size;
    }
    bias_c = (real *)malloc(sizeof(real) * (vocab_size2 + 2));
    for (b = 0; b < vocab_size2; ++b) {
        bias_c[b + 1] = (rand() / (real)RAND_MAX - 0.5) / vector_size;
    }
    bias_w2 = (real *)malloc(sizeof(real) * (vocab_size1 + 2));
    for (b = 0; b < vocab_size1; ++b) {
        bias_w2[b + 1] = (rand() / (real)RAND_MAX - 0.5) / vector_size;
    }
    bias_c2 = (real *)malloc(sizeof(real) * (vocab_size2 + 2));
    for (b = 0; b < vocab_size2; ++b) {
        bias_c2[b + 1] = (rand() / (real)RAND_MAX - 0.5) / vector_size;
    }
    for (b = 0; b < vector_size; b++) for (a = 0; a < 2 * vocab_size1; a++) W1[a * vector_size + b] = (rand() / (real)RAND_MAX - 0.5) / vector_size;
    for (b = 0; b < vector_size; b++) for (a = 0; a < 2 * vocab_size2; a++) W2[a * vector_size + b] = (rand() / (real)RAND_MAX - 0.5) / vector_size;

    vector_size--;
}

/* Train the GloVe model */
void *glove_thread(void *vid) {
    long long a, b ,l1, l2, cur_dev;
    int i, i2, cur_matrix, cur_word, cur_word2, negative_start, negative_end;
    long long id = (long long) vid;
    real *W, *C, *W_bi, *C_bi, *b_w, *b_c;
    real cur_val;
    real diff, fdiff, temp1, temp2, mb;
    real rate, x_max;
    int index, tmp_index;
    cost[id] = 0;

    int *i_tmp;
    real *d_tmp;
    real *pmi_tmp;

    for (i = start_per_thread[id]; i < start_per_thread[id + 1]; ++i) {
      cur_word = thread_split[i];
      cur_matrix = thread_split_m[i];
      if (cur_matrix == 1) a = vocab_size1;
      else a = vocab_size2;
      for (i2 = 1; i2 < a + 1; ++i2) positive[id][i2] = negative_val;
      for (i2 = 1; i2 < a + 1; ++i2) pmi[id][i2] = negative_pmi;
      for (i2 = 1; i2 < a + 1; ++i2) positions[id][i2] = i2;
      FILE *find, *finpmi, *fini;
      if (cur_matrix == 1) {
        /*find = fopen(input_file_d1, "rb");
        finpmi = fopen(input_file_pmi1, "rb");
        fini = fopen(input_file_i1, "rb");*/
        i_tmp = buffer_i1;
        d_tmp = buffer_d1;
        pmi_tmp = buffer_pmi1;
        cur_dev = deviation1[cur_word];
        negative_start = deviation1[cur_word + 1] - cur_dev + 1;
      } else if (cur_matrix == 2) {
        /*find = fopen(input_file_d2, "rb");
        fini = fopen(input_file_i2, "rb");
        finpmi = fopen(input_file_pmi2, "rb");*/
        i_tmp = buffer_i2;
        d_tmp = buffer_d2;
        pmi_tmp = buffer_pmi2;
        cur_dev = deviation2[cur_word];
        negative_start = deviation2[cur_word + 1] - cur_dev + 1;
      } else {
        /*find = fopen(input_file_d_bi, "rb");
        fini = fopen(input_file_i_bi, "rb");
        finpmi = fopen(input_file_pmi_bi, "rb");*/
        i_tmp = buffer_i_bi;
        d_tmp = buffer_d_bi;
        pmi_tmp = buffer_pmi_bi;
        cur_dev = deviation_bi[cur_word];
        negative_start = deviation_bi[cur_word + 1] - cur_dev + 1;
      }
      d_tmp += cur_dev;
      i_tmp += cur_dev;
      pmi_tmp += cur_dev;


      tmp_index = 1;
      for (i2 = 1; i2 < negative_start; ++i2) {
        index = i_tmp[i2 - 1];
        positive[id][index] = d_tmp[i2 - 1];
        pmi[id][index] = pmi_tmp[i2 - 1];
        if (pmi[id][index] < min_pmi) {

        } else {
            swap_int(&positions[id][tmp_index], &positions[id][index]);
            ++tmp_index;
        }
      }
      negative_start = tmp_index;
      
      if (cur_matrix == 3) {
        negative_end = negative_start + negative_ratio_bi * vocab_size1;
      } else if (cur_matrix == 1) {
        negative_end = negative_start + negative_ratio * vocab_size1;
      } else {
        negative_end = negative_start + negative_ratio * vocab_size2;
      }
      if (cur_matrix == 1 || cur_matrix == 3) {
        if (negative_end > vocab_size1 + 1)
            negative_end = vocab_size1 + 1;
        for (i2 = negative_start; i2 < negative_end; ++i2) {
            swap_int(&positions[id][i2], &positions[id][rand_long(vocab_size1 + 1 - i2) + i2]);
        }
      } else {
        if (negative_end > vocab_size2 + 1)
            negative_end = vocab_size2 + 1;
        for (i2 = negative_start; i2 < negative_end; ++i2) {
            swap_int(&positions[id][i2], &positions[id][rand_long(vocab_size2 + 1 - i2) + i2]);
        }
      }
      
      for (i2 = 1; i2 < negative_end; ++i2) {
        swap_int(&positions[id][i2], &positions[id][rand_long(negative_end - i2) + i2]);
      }
      
      for (i2 = 1; i2 < negative_end; ++i2) {
        // Get location of words in W & gradsq
        cur_word2 = positions[id][i2];

        if (cur_matrix == 1) {
            W = W1 + (cur_word - 1LL) * (vector_size + 1);
            C = W1 + ((cur_word2 - 1LL) + vocab_size1) * (vector_size + 1);
            mb = matrixbias1;
            rate = eta * weight1;
            x_max = x_max1;
            b_w = W + vector_size;
            b_c = C + vector_size;
        } else if (cur_matrix == 2) {
            W = W2 + (cur_word - 1LL) * (vector_size + 1);
            C = W2 + ((cur_word2 - 1LL) + vocab_size2) * (vector_size + 1);
            mb = matrixbias2;
            rate = eta * weight2;
            x_max = x_max2;
            b_w = W + vector_size;
            b_c = C + vector_size;
        } else if (cur_matrix == 3) {
            W = W1 + (cur_word - 1LL) * (vector_size + 1);
            C = W2 + (cur_word2 - 1LL) * (vector_size + 1);
            b_w = bias_w + cur_word;
            b_c = bias_c + cur_word2;
            mb = matrixbias_bi;
            rate = eta * weightbi;
            x_max = x_max_bi;
        }

        // Calculate cost, save diff for gradients
        diff = 0;
        for(b = 0; b < vector_size; b++) diff += W[b] * C[b]; // dot product of word and context word vector
            
        diff += *b_w + *b_c + mb - log(positive[id][cur_word2]); // add separate bias for each word
        fdiff = (positive[id][cur_word2] > x_max) ? diff : pow(positive[id][cur_word2] / x_max, alpha) * diff; // multiply weighting function (f) with diff
        cost[id] += 0.5 * fdiff * diff; // weighted squared error
          
        // Adaptive gradient updates
        fdiff *= rate; // for ease in calculating gradient
        for(b = 0; b < vector_size; b++) {
        // learning rate times gradient for word vectors
            temp1 = fdiff * C[b];
            temp2 = fdiff * W[b];
        // adaptive updates
            W[b] -= (temp1 + lambda * W[b]);
            C[b] -= (temp2 + lambda * C[b]);
        }

        // updates for bias terms
        *b_w -= (fdiff + lambda * *b_w);
        *b_c -= (fdiff + lambda * *b_c);

        if (cur_matrix == 1) {
            matrixbias1 -= (fdiff + lambda * matrixbias1);
        } else if (cur_matrix == 2) {
            matrixbias2 -= (fdiff + lambda * matrixbias2);
        } else {
            matrixbias_bi -= (fdiff + lambda * matrixbias_bi);
        }
      }
    }
    
    pthread_exit(NULL);
}

/* Save params to file */
int save_params(char *save_W_file, real *W, long long vocab_size, char *vocab_file) {
    long long a, b;
    real c;
    char format[20];
    char output_file[MAX_STRING_LENGTH], output_file_gsq[MAX_STRING_LENGTH];
    char *word = malloc(sizeof(char) * MAX_STRING_LENGTH);
    FILE *fid, *fout, *fgs;
    
    if(use_binary > 0) { // Save parameters in binary file
        sprintf(output_file,"%s.bin",save_W_file);
        fout = fopen(output_file,"wb");
        if(fout == NULL) {fprintf(stderr, "Unable to open file %s.\n",save_W_file); return 1;}
        
        fid = fopen(vocab_file, "r");
        sprintf(format,"%%%ds",MAX_STRING_LENGTH);
        if(fid == NULL) {fprintf(stderr, "Unable to open file %s.\n",vocab_file); return 1;}
        
        fprintf(fout, "%lld %d\n", vocab_size, vector_size);
        for(a = 0; a < vocab_size; a++) {
            if(fscanf(fid,format,word) == 0) return 1;
        fprintf(fout, "%s ", word);
        for(b = 0; b < vector_size; b++) {
          c = W[a * (vector_size + 1) + b];
          fwrite(&c, sizeof(real), 1,fout);
          }
        for(b = 0; b < vector_size; b++) {
          c = W[(vocab_size + a) * (vector_size + 1) + b];
          fwrite(&c, sizeof(real), 1,fout);
          }
        fprintf(fout, "\n");
        if(fscanf(fid,format,word) == 0) return 1; // Eat irrelevant frequency entry
    }
        
        fclose(fout);
        fclose(fid);
    }
    if(use_binary != 1) { // Save parameters in text file
        sprintf(output_file,"%s.txt",save_W_file);
        fout = fopen(output_file,"wb");
        if(fout == NULL) {fprintf(stderr, "Unable to open file %s.\n",save_W_file); return 1;}
        fid = fopen(vocab_file, "r");
        sprintf(format,"%%%ds",MAX_STRING_LENGTH);
        if(fid == NULL) {fprintf(stderr, "Unable to open file %s.\n",vocab_file); return 1;}
        for(a = 0; a < vocab_size; a++) {
            if(fscanf(fid,format,word) == 0) return 1;
            fprintf(fout, "%s",word);
            if(model == 0) { // Save all parameters (including bias)
                for(b = 0; b < (vector_size + 1); b++) fprintf(fout," %lf", W[a * (vector_size + 1) + b]);
                for(b = 0; b < (vector_size + 1); b++) fprintf(fout," %lf", W[(vocab_size + a) * (vector_size + 1) + b]);
            }
            if(model == 1) // Save only "word" vectors (without bias)
                for(b = 0; b < vector_size; b++) fprintf(fout," %lf", W[a * (vector_size + 1) + b]);
            if(model == 2) // Save "word + context word" vectors (without bias)
                for(b = 0; b < vector_size; b++) fprintf(fout," %lf", W[a * (vector_size + 1) + b] + W[(vocab_size + a) * (vector_size + 1) + b]);
            fprintf(fout,"\n");
            if(fscanf(fid,format,word) == 0) return 1; // Eat irrelevant frequency entry
        }
        fclose(fid);
        fclose(fout);
    }
    return 0;
}

/* Train model */
int train_glove() {
    long long a, file_size = 0;
    int b, i;
    FILE *fin;
    real total_cost = 0;
    fprintf(stderr, "TRAINING MODEL\n");
    
    fin = fopen(input_file_d1, "rb");
    if(fin == NULL) {fprintf(stderr,"Unable to open cooccurrence file %s.\n",input_file_d1); return 1;}
    fseeko(fin, 0, SEEK_END);
    file_size += ftello(fin);
    fclose(fin);
    fin = fopen(input_file_d2, "rb");
    if(fin == NULL) {fprintf(stderr,"Unable to open cooccurrence file %s.\n",input_file_d2); return 1;}
    fseeko(fin, 0, SEEK_END);
    file_size += ftello(fin);
    fclose(fin);
    fin = fopen(input_file_d_bi, "rb");
    if(fin == NULL) {fprintf(stderr,"Unable to open cooccurrence file %s.\n",input_file_d_bi); return 1;}
    fseeko(fin, 0, SEEK_END);
    file_size += ftello(fin);
    fclose(fin);
    num_lines = file_size / (sizeof(real)); // Assuming the file isn't corrupt and consists only of CREC's
    
    fprintf(stderr,"Read %lld lines.\n", num_lines);
    
    if(verbose > 1) fprintf(stderr,"Initializing parameters...");
    initialize_parameters();
    if(verbose > 1) fprintf(stderr,"done.\n");
    if(verbose > 0) fprintf(stderr,"vector size: %d\n", vector_size);
    if(verbose > 0) fprintf(stderr,"vocab size1: %lld\n", vocab_size1);
    if(verbose > 0) fprintf(stderr,"vocab size2: %lld\n", vocab_size2);
    if(verbose > 0) fprintf(stderr,"x_max1: %lf\n", x_max1);
    if(verbose > 0) fprintf(stderr,"x_max2: %lf\n", x_max2);
    if(verbose > 0) fprintf(stderr,"x_max_bi: %lf\n", x_max_bi);
    if(verbose > 0) fprintf(stderr,"alpha: %lf\n", alpha);
    if(verbose > 0) fprintf(stderr,"lambda: %lf\n", lambda);
    if(verbose > 0) fprintf(stderr,"eta: %lf\n", eta);
    if(verbose > 0) fprintf(stderr,"negative-ratio: %lf\n", negative_ratio);
    if(verbose > 0) fprintf(stderr,"negative-ratio-bi: %lf\n", negative_ratio_bi);
    if(verbose > 0) fprintf(stderr,"negative-val: %lf\n", negative_val);
    if(verbose > 0) fprintf(stderr,"negative-pmi: %lf\n", negative_pmi);
    if(verbose > 0) fprintf(stderr,"weight1: %lf\n", weight1);
    if(verbose > 0) fprintf(stderr,"weight2: %lf\n", weight2);
    if(verbose > 0) fprintf(stderr,"weightbi: %lf\n", weightbi);
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    start_per_thread = (int *)malloc((num_threads + 1) * sizeof(int));
    thread_split = (int *)malloc((total_size + 1) * sizeof(int));
    thread_split_m = (int *)malloc((total_size + 1) * sizeof(int));

    a = (int)(vocab_size1 * sample1);
    b = 1;
    for (i = 0; i < a; ++i) {
        thread_split[b + i] = i % vocab_size1 + 1;
        thread_split_m[b + i] = 1;
    }
    a = (int)(vocab_size2 * sample2);
    b = 1 + (int)(vocab_size1 * sample1);
    for (i = 0; i < a; ++i) {
        thread_split[i + b] = i % vocab_size2 + 1;
        thread_split_m[i + b] = 2;
    }
    a = (int)(vocab_size1 * samplebi);
    b = 1 + (int)(vocab_size1 * sample1) + (int)(vocab_size2 * sample2);
    for (i = 0; i < a; ++i) {
        thread_split[i + b] = i % vocab_size1 + 1;
        thread_split_m[i + b] = 3;
    }

    for (a = 0; a < num_threads; a++) start_per_thread[a] = (total_size / num_threads) * a + 1;
    start_per_thread[a] = total_size + 1;

    init_eta = eta;
    init_lambda = lambda;
    // Lock-free asynchronous SGD
    for(b = 0; b < num_iter; b++) {
        eta = init_eta;
        lambda = eta * init_lambda;
        total_cost = 0;

        for (i = 1; i < total_size + 1; ++i) {
            a = rand_long(total_size + 1 - i);
            swap_int(&thread_split[i], &thread_split[a + i]);
            swap_int(&thread_split_m[i], &thread_split_m[a + i]);
        }

        for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, glove_thread, (void *)a);
            for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
            for (a = 0; a < num_threads; a++) total_cost += cost[a];
            fprintf(stderr,"iter: %03d, cost: %lf\n", b+1, total_cost/num_lines);
            if (isnan(total_cost))
                return -1;
    }

    return save_params(save_W1_file, W1, vocab_size1, vocab_file1) | save_params(save_W2_file, W2, vocab_size2, vocab_file2);
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

void load_file(char *file_name, char **buffer) {
    FILE *fid;
    long long size;
    fid = fopen(file_name, "r");
    fseeko(fid, 0, SEEK_END);
    size = ftello(fid);
    fseeko(fid, 0, SEEK_SET);
    *buffer = malloc(size);
    fread(*buffer, 1, size, fid);
    fclose(fid);
}

int main(int argc, char **argv) {
    int i;
    FILE *fid;
    vocab_file1 = malloc(sizeof(char) * MAX_STRING_LENGTH);
    vocab_file2 = malloc(sizeof(char) * MAX_STRING_LENGTH);
    input_file1 = malloc(sizeof(char) * MAX_STRING_LENGTH);
    input_file2 = malloc(sizeof(char) * MAX_STRING_LENGTH);
    input_file_bi = malloc(sizeof(char) * MAX_STRING_LENGTH);
    save_W1_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
    save_W2_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
    input_file_p1 = malloc(sizeof(char) * MAX_STRING_LENGTH);
    input_file_i1 = malloc(sizeof(char) * MAX_STRING_LENGTH);
    input_file_d1 = malloc(sizeof(char) * MAX_STRING_LENGTH);
    input_file_pmi1 = malloc(sizeof(char) * MAX_STRING_LENGTH);
    input_file_p2 = malloc(sizeof(char) * MAX_STRING_LENGTH);
    input_file_i2 = malloc(sizeof(char) * MAX_STRING_LENGTH);
    input_file_d2 = malloc(sizeof(char) * MAX_STRING_LENGTH);
    input_file_pmi2 = malloc(sizeof(char) * MAX_STRING_LENGTH);
    input_file_p_bi = malloc(sizeof(char) * MAX_STRING_LENGTH);
    input_file_i_bi = malloc(sizeof(char) * MAX_STRING_LENGTH);
    input_file_d_bi = malloc(sizeof(char) * MAX_STRING_LENGTH);
    input_file_pmi_bi = malloc(sizeof(char) * MAX_STRING_LENGTH);
    
    if (argc == 1) {
        printf("Cross-lingual word embeddings via matrix co-factorization\n");
        printf("based on the GloVe package\n");
        printf("Revised by: Tianze Shi\n");
        printf("Original Author: Jeffrey Pennington\n\n");
        printf("Usage options:\n");
        printf("\t-verbose <int>\n");
        printf("\t\tSet verbosity: 0, 1, or 2 (default)\n");
        printf("\t-vector-size <int>\n");
        printf("\t\tDimension of word vector representations (excluding bias term); default 50\n");
        printf("\t-threads <int>\n");
        printf("\t\tNumber of threads; default 8\n");
        printf("\t-iter <int>\n");
        printf("\t\tNumber of training iterations; default 25\n");
        printf("\t-eta <float>\n");
        printf("\t\tInitial learning rate; default 0.05\n");
        printf("\t-alpha <float>\n");
        printf("\t\tParameter in exponent of weighting function; default 0.75\n");
        printf("\t-x-max[,2,bi] <float>\n");
        printf("\t\tParameter specifying cutoff in weighting function; default 100.0\n");
        printf("\t-binary <int>\n");
        printf("\t\tSave output in binary format (0: text, 1: binary, 2: both); default 0\n");
        printf("\t-input-file[1,2,-bi] <file>\n");
        printf("\t\tBinary input file of shuffled cooccurrence data (produced by 'cooccur' and 'shuffle'); default cooccurrence.shuf.bin\n");
        printf("\t-vocab-file[1,2] <file>\n");
        printf("\t\tFile containing vocabulary (truncated unigram counts, produced by 'vocab_count'); default vocab.txt\n");
        printf("\t-save-file[1,2] <file>\n");
        printf("\t\tFilename, excluding extension, for word vector output; default vectors\n");
        printf("\t-weight[1,2] <file>\n");
        printf("\t\tWeight of different matrices\n");
        return 0;
    }
    
    
    if ((i = find_arg((char *)"-verbose", argc, argv)) > 0) verbose = atoi(argv[i + 1]);
    if ((i = find_arg((char *)"-vector-size", argc, argv)) > 0) vector_size = atoi(argv[i + 1]);
    if ((i = find_arg((char *)"-iter", argc, argv)) > 0) num_iter = atoi(argv[i + 1]);
    if ((i = find_arg((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    cost = malloc(sizeof(real) * num_threads);
    if ((i = find_arg((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
    if ((i = find_arg((char *)"-negative-ratio", argc, argv)) > 0) negative_ratio = atof(argv[i + 1]);
    if ((i = find_arg((char *)"-negative-ratio-bi", argc, argv)) > 0) negative_ratio_bi = atof(argv[i + 1]);
    if ((i = find_arg((char *)"-x-max", argc, argv)) > 0) x_max1 = atof(argv[i + 1]);
    if ((i = find_arg((char *)"-x-max2", argc, argv)) > 0) x_max2 = atof(argv[i + 1]);
    else x_max2 = x_max1;
    if ((i = find_arg((char *)"-x-max-bi", argc, argv)) > 0) x_max_bi = atof(argv[i + 1]);
    else x_max_bi = x_max1;
    if ((i = find_arg((char *)"-eta", argc, argv)) > 0) eta = atof(argv[i + 1]);
    if ((i = find_arg((char *)"-negative-val", argc, argv)) > 0) negative_val = atof(argv[i + 1]);
    else negative_val= -x_max1;
    if ((i = find_arg((char *)"-negative-pmi", argc, argv)) > 0) negative_pmi = strtof(argv[i + 1], NULL);
    if ((i = find_arg((char *)"-min-pmi", argc, argv)) > 0) min_pmi = strtof(argv[i + 1], NULL);
    if ((i = find_arg((char *)"-lambda", argc, argv)) > 0) lambda = atof(argv[i + 1]);
    if ((i = find_arg((char *)"-weight1", argc, argv)) > 0) weight1 = atof(argv[i + 1]);
    if ((i = find_arg((char *)"-weight2", argc, argv)) > 0) weight2 = atof(argv[i + 1]);
    if ((i = find_arg((char *)"-weightbi", argc, argv)) > 0) weightbi = atof(argv[i + 1]);
    if ((i = find_arg((char *)"-sample1", argc, argv)) > 0) sample1 = atof(argv[i + 1]);
    if ((i = find_arg((char *)"-sample2", argc, argv)) > 0) sample2 = atof(argv[i + 1]);
    if ((i = find_arg((char *)"-samplebi", argc, argv)) > 0) samplebi = atof(argv[i + 1]);
    if ((i = find_arg((char *)"-binary", argc, argv)) > 0) use_binary = atoi(argv[i + 1]);
    if ((i = find_arg((char *)"-dump-all", argc, argv)) > 0) dump_all = atoi(argv[i + 1]);
    if ((i = find_arg((char *)"-model", argc, argv)) > 0) model = atoi(argv[i + 1]);
    if(model != 0 && model != 1) model = 2;
    if ((i = find_arg((char *)"-vocab-file1", argc, argv)) > 0) strcpy(vocab_file1, argv[i + 1]);
    else strcpy(vocab_file1, (char *)"vocab1.txt");
    if ((i = find_arg((char *)"-vocab-file2", argc, argv)) > 0) strcpy(vocab_file2, argv[i + 1]);
    else strcpy(vocab_file2, (char *)"vocab2.txt");
    if ((i = find_arg((char *)"-save-file1", argc, argv)) > 0) strcpy(save_W1_file, argv[i + 1]);
    else strcpy(save_W1_file, (char *)"vectors1");
    if ((i = find_arg((char *)"-save-file2", argc, argv)) > 0) strcpy(save_W2_file, argv[i + 1]);
    else strcpy(save_W2_file, (char *)"vectors2");
    if ((i = find_arg((char *)"-input-file1", argc, argv)) > 0) strcpy(input_file1, argv[i + 1]);
    else strcpy(input_file1, (char *)"cooccurrence.shuf.bin");
    if ((i = find_arg((char *)"-input-file2", argc, argv)) > 0) strcpy(input_file2, argv[i + 1]);
    else strcpy(input_file2, (char *)"cooccurrence.shuf.bin");
    if ((i = find_arg((char *)"-input-file-bi", argc, argv)) > 0) strcpy(input_file_bi, argv[i + 1]);
    else strcpy(input_file_bi, (char *)"cooccurrence.shuf.bin");

    vocab_size1 = 0;
    fid = fopen(vocab_file1, "r");
    if(fid == NULL) {fprintf(stderr, "Unable to open vocab file1 %s.\n",vocab_file1); return 1;}
    while ((i = getc(fid)) != EOF) if (i == '\n') vocab_size1++; // Count number of entries in vocab_file
    fclose(fid);
    
    vocab_size2 = 0;
    fid = fopen(vocab_file2, "r");
    if(fid == NULL) {fprintf(stderr, "Unable to open vocab file1 %s.\n",vocab_file2); return 1;}
    while ((i = getc(fid)) != EOF) if (i == '\n') vocab_size2++; // Count number of entries in vocab_file
    fclose(fid);

    total_size = vocab_size1 * 2 + vocab_size2;
    total_size = (int)(vocab_size1 * sample1) + (int)(vocab_size1 * samplebi) + (int)(vocab_size2 * sample2);
    
    deviation1 = (long long *)malloc(sizeof(long long) * (vocab_size1 + 2));
    sprintf(input_file_d1,"%s.d",input_file1);
    sprintf(input_file_i1,"%s.i",input_file1);
    sprintf(input_file_p1,"%s.p",input_file1);
    sprintf(input_file_pmi1,"%s.pmi",input_file1);
    fid = fopen(input_file_p1, "rb");
    if(fid == NULL) {fprintf(stderr, "Unable to open vocab file %s.\n",input_file_p1); return 1;}
    deviation1[0] = 0;
    deviation1[1] = 0;
    i = 1;
    while (!feof(fid)) {
      ++i;
      fread(&count, sizeof(long long), 1, fid);
      deviation1[i] = count;
    }

    deviation2 = (long long *)malloc(sizeof(long long) * (vocab_size2 + 2));
    sprintf(input_file_d2,"%s.d",input_file2);
    sprintf(input_file_i2,"%s.i",input_file2);
    sprintf(input_file_p2,"%s.p",input_file2);
    sprintf(input_file_pmi2,"%s.pmi",input_file2);
    fid = fopen(input_file_p2, "rb");
    if(fid == NULL) {fprintf(stderr, "Unable to open vocab file %s.\n",input_file_p2); return 1;}
    deviation2[0] = 0;
    deviation2[1] = 0;
    i = 1;
    while (!feof(fid)) {
      ++i;
      fread(&count, sizeof(long long), 1, fid);
      deviation2[i] = count;
    }

    deviation_bi = (long long *)malloc(sizeof(long long) * (vocab_size1 + 2));
    sprintf(input_file_d_bi,"%s.d",input_file_bi);
    sprintf(input_file_i_bi,"%s.i",input_file_bi);
    sprintf(input_file_p_bi,"%s.p",input_file_bi);
    sprintf(input_file_pmi_bi,"%s.pmi",input_file_bi);
    fid = fopen(input_file_p_bi, "rb");
    if(fid == NULL) {fprintf(stderr, "Unable to open vocab file %s.\n",input_file_p_bi); return 1;}
    deviation_bi[0] = 0;
    deviation_bi[1] = 0;
    i = 1;
    while (!feof(fid)) {
      ++i;
      fread(&count, sizeof(long long), 1, fid);
      deviation_bi[i] = count;
    }
    
    positions = (int **)malloc(sizeof(int *) * num_threads);
    positive = (real **)malloc(sizeof(real *) * num_threads);
    pmi = (real **)malloc(sizeof(real *) * num_threads);
    for (i = 0; i < num_threads; ++i) {
      positions[i] = (int *)malloc(sizeof(int) * (vocab_size1 * 2 + vocab_size2 + 1));
      positive[i] = (real *)malloc(sizeof(real) * (vocab_size1 + vocab_size2 + 1));
      pmi[i] = (real *)malloc(sizeof(real) * (vocab_size1 + vocab_size2 + 1));
    }

    load_file(input_file_d1, (char **)&buffer_d1);
    load_file(input_file_d2, (char **)&buffer_d2);
    load_file(input_file_d_bi, (char **)&buffer_d_bi);
    load_file(input_file_i1, (char **)&buffer_i1);
    load_file(input_file_i2, (char **)&buffer_i2);
    load_file(input_file_i_bi, (char **)&buffer_i_bi);  
    load_file(input_file_pmi1, (char **)&buffer_pmi1);
    load_file(input_file_pmi2, (char **)&buffer_pmi2);
    load_file(input_file_pmi_bi, (char **)&buffer_pmi_bi);   
    
    return train_glove();
}
