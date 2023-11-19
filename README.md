# Cross-lingual-Word-Embeddings
#Usage
  run make to compile
  run ./vocab-count.out < corpus.txt > vocab.txt to determine vocabulary (run once for each language), or you may use a given vocabulary from elsewhere.
  run ./cooccur-mono.out < corpus.txt > mono.cooc to count monolingual co-occurrences.
  run ./cooccur-clc-wa.out < corpus.txt > cross.cooc to count cross-lingual co-occurrences. (similar for ./cooccur-clc+wa.out < A3.final > cross.cooc. For CLSim, you need to convert t3.final into the same file format produced by cooccur*.out)
  run ./summarize-real.out -input-file mono.cooc -save-file mono.cooc to convert each co-occurrence file into sparse format and calculate PMI values.
  run ./mf-bi-clc.out -vocab-file1 vocab1.txt -vocab-file2 vocab2.txt -iter 30 -threads 20 -input-file1 mono1.cooc -input-file2 mono2.cooc -input-file-bi bi.cooc -vector-size 40 -binary 1 -save-file1 vectors1.bin -save-file2 vectors2.bin to learn cross-lingual embeddings. (similar for ./mf-bi-clsim)
