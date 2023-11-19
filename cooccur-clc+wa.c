//  Tool to calculate cross-lingual word-word cooccurrence statistics
//  (with sentence-aligned corpus, with word alignments)
//
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

#define TSIZE 1048576
#define SEED 1159241
#define HASHFN bitwisehash

static const int MAX_STRING_LENGTH = 1000;
typedef double real;

typedef struct cooccur_rec {
    int lan1;
    int lan2;
    int word1;
    int word2;
    real val;
} CREC;

typedef struct cooccur_rec_id {
    int lan1;
    int lan2;
    int word1;
    int word2;
    real val;
    int id;
} CRECID;

typedef struct hashrec {
    char	*word;
    long long id;
    struct hashrec *next;
} HASHREC;

int verbose = 2, lan1 = 1, lan2 = 2; // 0, 1, or 2
long long max_product; // Cutoff for product of word frequency ranks below which cooccurrence counts will be stored in a compressed full array
long long overflow_length; // Number of cooccurrence records whose product exceeds max_product to store in memory before writing to disk
real memory_limit = 3; // soft limit, in gigabytes, used to estimate optimal array sizes
char *vocab_file1, *vocab_file2, *file_head;
int window_size = 15; // default context window size
int model = 3;

/* Efficient string comparison */
int scmp( char *s1, char *s2 ) {
    while(*s1 != '\0' && *s1 == *s2) {s1++; s2++;}
    return(*s1 - *s2);
}

const long long min(const long long a, const long long b) {
    if (a < b)
        return a;
    else
        return b;
}

const long long max(const long long a, const long long b) {
    if (a < b)
        return b;
    else
        return a;
}

/* Move-to-front hashing and hash function from Hugh Williams, http://www.seg.rmit.edu.au/code/zwh-ipl/ */

/* Simple bitwise hash function */
unsigned int bitwisehash(char *word, int tsize, unsigned int seed) {
    char c;
    unsigned int h;
    h = seed;
    for(; (c =* word) != '\0'; word++) h ^= ((h << 5) + c + (h >> 2));
    return((unsigned int)((h&0x7fffffff) % tsize));
}

/* Create hash table, initialise pointers to NULL */
HASHREC ** inithashtable() {
    int	i;
    HASHREC **ht;
    ht = (HASHREC **) malloc( sizeof(HASHREC *) * TSIZE );
    for(i = 0; i < TSIZE; i++) ht[i] = (HASHREC *) NULL;
    return(ht);
}

/* Search hash table for given string, return record if found, else NULL */
HASHREC *hashsearch(HASHREC **ht, char *w) {
    HASHREC	*htmp, *hprv;
    unsigned int hval = HASHFN(w, TSIZE, SEED);
    for(hprv = NULL, htmp=ht[hval]; htmp != NULL && scmp(htmp->word, w) != 0; hprv = htmp, htmp = htmp->next);
    if( htmp != NULL && hprv!=NULL ) { // move to front on access
        hprv->next = htmp->next;
        htmp->next = ht[hval];
        ht[hval] = htmp;
    }
    return(htmp);
}

/* Insert string in hash table, check for duplicates which should be absent */
void hashinsert(HASHREC **ht, char *w, long long id) {
    HASHREC	*htmp, *hprv;
    unsigned int hval = HASHFN(w, TSIZE, SEED);
    for(hprv = NULL, htmp = ht[hval]; htmp != NULL && scmp(htmp->word, w) != 0; hprv = htmp, htmp = htmp->next);
    if(htmp == NULL) {
        htmp = (HASHREC *) malloc(sizeof(HASHREC));
        htmp->word = (char *) malloc(strlen(w) + 1);
        strcpy(htmp->word, w);
        htmp->id = id;
        htmp->next = NULL;
        if(hprv == NULL) ht[hval] = htmp;
        else hprv->next = htmp;
    }
    else fprintf(stderr, "Error, duplicate entry located: %s.\n",htmp->word);
    return;
}

/* Read word from input stream */
int get_word(char *word, FILE *fin) {
    int i = 0, ch;
    while(!feof(fin)) {
        ch = fgetc(fin);
        if(ch == 13) continue;
        if((ch == ' ') || (ch == '\t') || (ch == '\n')) {
            if(i > 0) {
                if (ch == '\n') ungetc(ch, fin);
                break;
            }
            if (ch == '\n') return 1;
            else continue;
        }
        word[i++] = ch;
        if(i >= MAX_STRING_LENGTH - 1) i--;   // truncate words that exceed max length
    }
    word[i] = 0;
    return 0;
}

/* Write sorted chunk of cooccurrence records to file, accumulating duplicate entries */
int write_chunk(CREC *cr, long long length, FILE *fout) {
    long long a = 0;
    CREC old = cr[a];
    
    for(a = 1; a < length; a++) {
        if(cr[a].word1 == old.word1 && cr[a].word2 == old.word2) {
            old.val += cr[a].val;
            continue;
        }
        fwrite(&old, sizeof(CREC), 1, fout);
        old = cr[a];
    }
    fwrite(&old, sizeof(CREC), 1, fout);
    return 0;
}

/* Check if two cooccurrence records are for the same two words, used for qsort */
int compare_crec(const void *a, const void *b) {
    int c;
    if( (c = ((CREC *) a)->word1 - ((CREC *) b)->word1) != 0) return c;
    else return (((CREC *) a)->word2 - ((CREC *) b)->word2);
    
}

/* Check if two cooccurrence records are for the same two words */
int compare_crecid(CRECID a, CRECID b) {
    int c;
    if( (c = a.word1 - b.word1) != 0) return c;
    else return a.word2 - b.word2;
}

/* Swap two entries of priority queue */
void swap_entry(CRECID *pq, int i, int j) {
    CRECID temp = pq[i];
    pq[i] = pq[j];
    pq[j] = temp;
}

/* Insert entry into priority queue */
void insert(CRECID *pq, CRECID new, int size) {
    int j = size - 1, p;
    pq[j] = new;
    while( (p=(j-1)/2) >= 0 ) {
        if(compare_crecid(pq[p],pq[j]) > 0) {swap_entry(pq,p,j); j = p;}
        else break;
    }
}

/* Delete entry from priority queue */
void delete(CRECID *pq, int size) {
    int j, p = 0;
    pq[p] = pq[size - 1];
    while( (j = 2*p+1) < size - 1 ) {
        if(j == size - 2) {
            if(compare_crecid(pq[p],pq[j]) > 0) swap_entry(pq,p,j);
            return;
        }
        else {
            if(compare_crecid(pq[j], pq[j+1]) < 0) {
                if(compare_crecid(pq[p],pq[j]) > 0) {swap_entry(pq,p,j); p = j;}
                else return;
            }
            else {
                if(compare_crecid(pq[p],pq[j+1]) > 0) {swap_entry(pq,p,j+1); p = j + 1;}
                else return;
            }
        }
    }
}

/* Write top node of priority queue to file, accumulating duplicate entries */
int merge_write(CRECID new, CRECID *old, FILE *fout) {
    if(new.word1 == old->word1 && new.word2 == old->word2) {
        old->val += new.val;
        return 0; // Indicates duplicate entry
    }
    fwrite(old, sizeof(CREC), 1, fout);
    *old = new;
    return 1; // Actually wrote to file
}

/* Merge [num] sorted files of cooccurrence records */
int merge_files(int num) {
    int i, size;
    long long counter = 0;
    CRECID *pq, new, old;
    char filename[200];
    FILE **fid, *fout;
    fid = malloc(sizeof(FILE) * num);
    pq = malloc(sizeof(CRECID) * num);
    fout = stdout;
    if(verbose > 1) fprintf(stderr, "Merging cooccurrence files: processed 0 lines.");
    
    /* Open all files and add first entry of each to priority queue */
    for(i = 0; i < num; i++) {
        sprintf(filename,"%s_%04d.bin",file_head,i);
        fid[i] = fopen(filename,"rb");
        if(fid[i] == NULL) {fprintf(stderr, "Unable to open file %s.\n",filename); return 1;}
        fread(&new, sizeof(CREC), 1, fid[i]);
        new.id = i;
        insert(pq,new,i+1);
    }
    
    /* Pop top node, save it in old to see if the next entry is a duplicate */
    size = num;
    old = pq[0];
    i = pq[0].id;
    delete(pq, size);
    fread(&new, sizeof(CREC), 1, fid[i]);
    if(feof(fid[i])) size--;
    else {
        new.id = i;
        insert(pq, new, size);
    }
    
    /* Repeatedly pop top node and fill priority queue until files have reached EOF */
    while(size > 0) {
        counter += merge_write(pq[0], &old, fout); // Only count the lines written to file, not duplicates
        if((counter%100000) == 0) if(verbose > 1) fprintf(stderr,"\033[39G%lld lines.",counter);
        i = pq[0].id;
        delete(pq, size);
        fread(&new, sizeof(CREC), 1, fid[i]);
        if(feof(fid[i])) size--;
        else {
            new.id = i;
            insert(pq, new, size);
        }
    }
    fwrite(&old, sizeof(CREC), 1, fout);
    fprintf(stderr,"\033[0GMerging cooccurrence files: processed %lld lines.\n",++counter);
    for(i=0;i<num;i++) {
        sprintf(filename,"%s_%04d.bin",file_head,i);
        remove(filename);
    }
    fprintf(stderr,"\n");
    return 0;
}

/* Collect word-word cooccurrence counts from input stream */
int get_cooccurrence() {
    int flag, x, y, fidcounter = 1, sent1_len = 0, sent2_len = 0, sent3_len = 0, read_mod = 0;
    long long a, j = 0, k, l, iter, id, counter = 0, ind = 0, vocab_size1, vocab_size2, w1, w2, *lookup, *sentence1, *sentence2, *sentence3;
    long long base1, base2, cur1, cur2;
    char format[20], filename[200], str[MAX_STRING_LENGTH + 1];
    FILE *fid, *foverflow;
    real *bigram_table, r;
    HASHREC *htmp, **vocab_hash1 = inithashtable(), **vocab_hash2 = inithashtable();
    CREC *cr = malloc(sizeof(CREC) * (overflow_length + 10000));
    sentence1 = malloc(sizeof(long long) * MAX_STRING_LENGTH);
    sentence2 = malloc(sizeof(long long) * MAX_STRING_LENGTH);
    sentence3 = malloc(sizeof(long long) * MAX_STRING_LENGTH);
    
    fprintf(stderr, "COUNTING COOCCURRENCES\n");

    if(verbose > 1) fprintf(stderr, "max product: %lld\n", max_product);
    if(verbose > 1) fprintf(stderr, "overflow length: %lld\n", overflow_length);
    sprintf(format,"%%%ds %%lld", MAX_STRING_LENGTH); // Format to read from vocab file, which has (irrelevant) frequency data
    
    // Reading vocab1
    if(verbose > 1) fprintf(stderr, "Reading vocab1 from file \"%s\"...", vocab_file1);
    fid = fopen(vocab_file1,"r");
    if(fid == NULL) {fprintf(stderr,"Unable to open vocab file %s.\n",vocab_file1); return 1;}
    while(fscanf(fid, format, str, &id) != EOF) hashinsert(vocab_hash1, str, ++j); // Here id is not used: inserting vocab words into hash table with their frequency rank, j
    fclose(fid);
    vocab_size1 = j;
    j = 0;
    if(verbose > 1) fprintf(stderr, "loaded %lld words.\nBuilding lookup table...", vocab_size1);
    
    // Reading vocab2
    if(verbose > 1) fprintf(stderr, "Reading vocab2 from file \"%s\"...", vocab_file2);
    fid = fopen(vocab_file2,"r");
    if(fid == NULL) {fprintf(stderr,"Unable to open vocab file %s.\n",vocab_file2); return 1;}
    while(fscanf(fid, format, str, &id) != EOF) hashinsert(vocab_hash2, str, ++j); // Here id is not used: inserting vocab words into hash table with their frequency rank, j
    fclose(fid);
    vocab_size2 = j;
    j = 0;
    if(verbose > 1) fprintf(stderr, "loaded %lld words.\nBuilding lookup table...", vocab_size2);
    
    /* Build auxiliary lookup table used to index into bigram_table */
    lookup = (long long *)calloc( vocab_size1 + 1 , sizeof(long long) );
    if (lookup == NULL) {
        fprintf(stderr, "Couldn't allocate memory!");
        return 1;
    }
    lookup[0] = 1;
    for(a = 1; a <= vocab_size1; a++) {
        if((lookup[a] = max_product / a) < vocab_size2) lookup[a] += lookup[a-1];
        else lookup[a] = lookup[a-1] + vocab_size2;
    }
    if(verbose > 1) fprintf(stderr, "table contains %lld elements.\n",lookup[a-1]);
    
    /* Allocate memory for full array which will store all cooccurrence counts for words whose product of frequency ranks is less than max_product */
    bigram_table = (real *)calloc( lookup[a-1] , sizeof(real) );
    if (bigram_table == NULL) {
        fprintf(stderr, "Couldn't allocate memory!");
        return 1;
    }
    
    fid = stdin;
    sprintf(format,"%%%ds",MAX_STRING_LENGTH);
    sprintf(filename,"%s_%04d.bin",file_head, fidcounter);
    foverflow = fopen(filename,"w");
    if(verbose > 1) fprintf(stderr,"Processing token: 0");
    
    /* For each token in input stream, calculate a cooccurrence sum*/
    while (1) {
	flag = get_word(str, fid);
	if(feof(fid)) break;
	if(flag == 1) {		// Newline

	    if (read_mod == 0)
            read_mod = 1;
        else if (read_mod == 1)
            read_mod = 2;
	    else {
		// Calculate Co-occurrence
        for (j = 0; j < sent3_len; j = j + 2) {
            base1 = sentence3[j];
            base2 = sentence3[j + 1];

            w1 = sentence1[base1];
            w2 = sentence2[base2];

            if (w1 < 0 || w2 < 0)
                continue;

            if ( w1 < max_product/w2 ) { // Product is small enough to store in a full array
                bigram_table[lookup[w1 - 1] + w2 - 2] += 1.0; // Weight by inverse of distance between words
            }
            else { // Product is too big, data is likely to be sparse. Store these entries in a temporary buffer to be sorted, merged (accumulated), and written to file when it gets full.
                cr[ind].lan1 = lan1;
                cr[ind].lan2 = lan2;
                cr[ind].word1 = w1;
                cr[ind].word2 = w2;
                cr[ind].val = 1.0;
                ind++; // Keep track of how full temporary buffer is
            }


            for (iter = 0; iter < sent3_len; iter = iter + 2) {
                if (iter == j) continue;

                cur1 = sentence3[iter];
                cur2 = sentence3[iter + 1];

                if (cur1 - base1 > window_size || base1 - cur1 > window_size) continue;
                if (cur2 - base2 > window_size || base2 - cur2 > window_size) continue;

                if (model == 2 || model == 3) {
                k = cur1;
                w1 = sentence1[k];
                w2 = sentence2[base2];

                if (w1 < 0 || w2 < 0)
                    continue;

    			if ( w1 < max_product/w2 ) { // Product is small enough to store in a full array
    			    bigram_table[lookup[w1 - 1] + w2 - 2] += 1.0/(fabs(k-base1) + 1); // Weight by inverse of distance between words
    			}
    			else { // Product is too big, data is likely to be sparse. Store these entries in a temporary buffer to be sorted, merged (accumulated), and written to file when it gets full.
    			    cr[ind].lan1 = lan1;
    			    cr[ind].lan2 = lan2;
    			    cr[ind].word1 = w1;
    			    cr[ind].word2 = w2;
                    cr[ind].val = 1.0/(fabs(k-base1) + 1);
    			    ind++; // Keep track of how full temporary buffer is
    			}
                
                }

                if (model == 1 || model == 3) {
                k = cur2;
                if (model == 3 && k == base2) continue;
                w1 = sentence1[base1];
                w2 = sentence2[k];

                if (w1 < 0 || w2 < 0)
                    continue;

                if ( w1 < max_product/w2 ) { // Product is small enough to store in a full array
                    bigram_table[lookup[w1 - 1] + w2 - 2] += 1.0/(fabs(k-base2) + 1); // Weight by inverse of distance between words
                }
                else { // Product is too big, data is likely to be sparse. Store these entries in a temporary buffer to be sorted, merged (accumulated), and written to file when it gets full.
                    cr[ind].lan1 = lan1;
                    cr[ind].lan2 = lan2;
                    cr[ind].word1 = w1;
                    cr[ind].word2 = w2;
                    cr[ind].val = 1.0/(fabs(k-base2) + 1);
                    ind++; // Keep track of how full temporary buffer is
                }
                
                }

            }
        }
		
		
		read_mod = 0;
		sent1_len = 0;
		sent2_len = 0;
        sent3_len = 0;
		// If overflow buffer is (almost) full, sort it and write it to temporary file
		if(ind >= overflow_length) {
		    qsort(cr, ind, sizeof(CREC), compare_crec);
		    write_chunk(cr,ind,foverflow);
		    fclose(foverflow);
		    fidcounter++;
		    sprintf(filename,"%s_%04d.bin",file_head,fidcounter);
		    foverflow = fopen(filename,"w");
		    ind = 0;
		}
	    }
	} else {
	
	counter++;
	if((counter%100000) == 0) if(verbose > 1) fprintf(stderr,"\033[19G%lld",counter);
	if (read_mod == 0) {
	    htmp = hashsearch(vocab_hash1, str);
	    if (htmp == NULL) sentence1[sent1_len++] = -1;
        else sentence1[sent1_len++] = htmp->id;
	} else if (read_mod == 1) {
	    htmp = hashsearch(vocab_hash2, str);
	    if (htmp == NULL) sentence2[sent2_len++] = -1;
        else sentence2[sent2_len++] = htmp->id;
	} else if (read_mod == 2) {
        sentence3[sent3_len++] = atoi(str);
    }
    }
    }
    
    /* Write out temp buffer for the final time (it may not be full) */
    if(verbose > 1) fprintf(stderr,"\033[0GProcessed %lld tokens.\n",counter);
    qsort(cr, ind, sizeof(CREC), compare_crec);
    write_chunk(cr,ind,foverflow);
    sprintf(filename,"%s_0000.bin",file_head);
    
    /* Write out full bigram_table, skipping zeros */
    if(verbose > 1) fprintf(stderr, "Writing cooccurrences to disk");
    fid = fopen(filename,"w");
    j = 1e6;
    for(x = 1; x <= vocab_size1; x++) {
        if( (long long) (0.75*log(vocab_size1 / x)) < j) {j = (long long) (0.75*log(vocab_size1 / x)); if(verbose > 1) fprintf(stderr,".");} // log's to make it look (sort of) pretty
        for(y = 1; y <= (lookup[x] - lookup[x-1]); y++) {
            if((r = bigram_table[lookup[x-1] - 2 + y]) != 0) {
		fwrite(&lan1, sizeof(int), 1, fid);
		fwrite(&lan2, sizeof(int), 1, fid);
                fwrite(&x, sizeof(int), 1, fid);
                fwrite(&y, sizeof(int), 1, fid);
                fwrite(&r, sizeof(real), 1, fid);
            }
        }
    }
    
    if(verbose > 1) fprintf(stderr,"%d files in total.\n",fidcounter + 1);
    fclose(fid);
    fclose(foverflow);
    free(cr);
    free(lookup);
    free(bigram_table);
    free(vocab_hash1);
    free(vocab_hash2);
    return merge_files(fidcounter + 1); // Merge the sorted temporary files
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

int main(int argc, char **argv) {
    int i;
    real rlimit, n = 1e5;
    vocab_file1 = malloc(sizeof(char) * MAX_STRING_LENGTH);
    vocab_file2 = malloc(sizeof(char) * MAX_STRING_LENGTH);
    file_head = malloc(sizeof(char) * MAX_STRING_LENGTH);
    
    if (argc == 1) {
        printf("Tool to calculate word-word cooccurrence statistics\n");
        printf("Originally distributed in the GloVe package\n");
        printf("Revised by: Tianze Shi\n");
        printf("Original Author: Jeffrey Pennington\n\n");
        printf("Usage options:\n");
        printf("\t-verbose <int>\n");
        printf("\t\tSet verbosity: 0, 1, or 2 (default)\n");
        printf("\t-window-size <int>\n");
        printf("\t\tNumber of context words to the left (and to the right, if symmetric = 1); default 15\n");
        printf("\t-vocab-file1 <file>\n");
        printf("\t\tFile containing vocabulary for language 1 (truncated unigram counts, produced by 'vocab_count'); default vocab.txt\n");
        printf("\t-vocab-file2 <file>\n");
        printf("\t\tFile containing vocabulary for language 2 (truncated unigram counts, produced by 'vocab_count'); default vocab.txt\n");
        printf("\t-lan1 <file>\n");
        printf("\t\tSet language id 1 (default is 1)\n");
        printf("\t-lan2 <file>\n");
        printf("\t\tSet language id 2 (default is 2)\n");
        printf("\t-memory <float>\n");
        printf("\t\tSoft limit for memory consumption, in GB -- based on simple heuristic, so not extremely accurate; default 4.0\n");
        printf("\t-max-product <int>\n");
        printf("\t\tLimit the size of dense cooccurrence array by specifying the max product <int> of the frequency counts of the two cooccurring words.\n\t\tThis value overrides that which is automatically produced by '-memory'. Typically only needs adjustment for use with very large corpora.\n");
        printf("\t-overflow-length <int>\n");
        printf("\t\tLimit to length <int> the sparse overflow array, which buffers cooccurrence data that does not fit in the dense array, before writing to disk. \n\t\tThis value overrides that which is automatically produced by '-memory'. Typically only needs adjustment for use with very large corpora.\n");
        printf("\t-overflow-file <file>\n");
        printf("\t\tFilename, excluding extension, for temporary files; default overflow\n");

        printf("\nExample usage:\n");
        printf("./cooccur -verbose 2 -vocab-file vocab.txt -memory 8.0 -overflow-file tempoverflow < corpus.txt > cooccurrences.bin\n\n");
        return 0;
    }

    if ((i = find_arg((char *)"-verbose", argc, argv)) > 0) verbose = atoi(argv[i + 1]);
    if ((i = find_arg((char *)"-lan1", argc, argv)) > 0) lan1 = atoi(argv[i + 1]);
    if ((i = find_arg((char *)"-lan2", argc, argv)) > 0) lan2 = atoi(argv[i + 1]);
    if ((i = find_arg((char *)"-model", argc, argv)) > 0) model = atoi(argv[i + 1]);
    if ((i = find_arg((char *)"-window-size", argc, argv)) > 0) window_size = atoi(argv[i + 1]);
    if ((i = find_arg((char *)"-vocab-file1", argc, argv)) > 0) strcpy(vocab_file1, argv[i + 1]);
    else strcpy(vocab_file1, (char *)"vocab.txt");
    if ((i = find_arg((char *)"-vocab-file2", argc, argv)) > 0) strcpy(vocab_file2, argv[i + 1]);
    else strcpy(vocab_file2, (char *)"vocab.txt");
    if ((i = find_arg((char *)"-overflow-file", argc, argv)) > 0) strcpy(file_head, argv[i + 1]);
    else strcpy(file_head, (char *)"overflow");
    if ((i = find_arg((char *)"-memory", argc, argv)) > 0) memory_limit = atof(argv[i + 1]);
    
    /* The memory_limit determines a limit on the number of elements in bigram_table and the overflow buffer */
    /* Estimate the maximum value that max_product can take so that this limit is still satisfied */
    rlimit = 0.85 * (real)memory_limit * 1073741824/(sizeof(CREC));
    while(fabs(rlimit - n * (log(n) + 0.1544313298)) > 1e-3) n = rlimit / (log(n) + 0.1544313298);
    max_product = (long long) n;
    overflow_length = (long long) rlimit/6; // 0.85 + 1/6 ~= 1
    
    /* Override estimates by specifying limits explicitly on the command line */
    if ((i = find_arg((char *)"-max-product", argc, argv)) > 0) max_product = atoll(argv[i + 1]);
    if ((i = find_arg((char *)"-overflow-length", argc, argv)) > 0) overflow_length = atoll(argv[i + 1]);
    
    return get_cooccurrence();
}


