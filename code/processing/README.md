# Processing
The scripts in this directory compute the various conversation metrics 
described in the paper. The names of the script map closely to the feature sets 
outlined in Table 1 and user/dyad/conversation characteristics analyzed in 
Sec 4.

Most of the scripts have three command-line options:
1. `--dataset`: news or midterms,
2. `--n_jobs`: number of workers to use when processing the data in parallel,
3. `--limit`: how many records/conversations to consider, useful for testing.

Note that due to the size of the datasets, most scripts take a while to 
complete. If you are only interested in their outputs, consult the data 
repository.
