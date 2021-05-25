# The Structure of Toxic Conversations on Twitter
[Martin Saveski](http://martinsaveski.com),
[Brandon Roy](http://alumni.media.mit.edu/~bcroy/), and 
[Deb Roy](https://www.media.mit.edu/people/dkroy/overview/)

This repository contains code for replicating the analysis and experiments in 
the paper, "[The Structure of Toxic Conversations on Twitter](https://doi.org/10.1145/3442381.3449861)" 
published in the proceedings of The Web Conference 2021 (WWW '21).

The anonymized data needed for replication is available at the 
[data repository hosted on the Harvard Dataverse](https://doi.org/10.7910/DVN/TRWPVB).
More details about the data under Data guide.


## Code guide
The code is split into four directories:
1. `processing`: which includes code that computes various metrics on the 
    conversations. Both the analyses (Sec 4) and the prediction models (Sec 5) 
    rely on the metrics implemented in these scripts. 
2. `analyses`: contains code for conducting the analyses of the conversations 
    after the conversations are over (Sec 4).
3. `modeling`: contains code for preparing the data for training the 
    classification models as well as training and evaluating the models (Sec 5).
4. `plotting`: includes the code for generating all the figures in the paper.

Before running any of the scripts, make sure to configure the paths to the 
data in `processing/_config.py` and install the Python packages in 
`requirements.txt`.

Most of the code is written in Python, except for the analyses and plotting 
code which was written in R.

Since the datasets are quite large, most scripts allow parallel computation.


### Analyses (Sec 4)
To generate the figures in the analyses section, run the following scripts for 
each of the two datasets (news and midterms). 
- _User analysis_ (Fig 2): `processing/user_metrics.py`, `analyses/1_users.R` 
    (NB: this analysis cannot be performed on the anonymized data, see the data 
    description section for more information.)  
- _Dyad analysis_ (Fig 3): `processing/dyad_metrics.py`, `analyses/2_dyads.R`.
- _Reply tree analysis_ (Figs 4 and 5): `processing/tree_metrics.py`, 
    `processing/toxicity_metrics.py`, `analyses/3_reply_trees.R`.
- _Follow graph analysis_ (Fig 6): `processing/reply_graph_metrics.py`, 
    `processing/toxicity_metrics.py`, `analyses/4_follow_graphs.R`.
- _Plotting_: finally, run `plotting/analyses.R` to generate the figures.

Note that computing some of the metrics takes awhile. The outputs of the scripts
is included in the data repository in `derived_data.7z`.


### Conversation toxicity predictions (Sec 5.1)
Run the following scripts:
1. `processing/prefix_metrics.py`: computes the metrics described in Table 1 at 
    various stages of the conversation (i.e., first 10, 20, ... 100 replies).
    This script relies on many of the other scripts in the `processing` 
    directory, each of which implements a different feature set.
2. `modeling/prefix_make_datasets.ipynb`: creates the design matrices (X) for 
    the prediction tasks for each prefix size. 
3. `modeling/prefix_make_labels.ipynb`: creates the outcome variables (y).
4. `modeling/prefix_prediction.py`: performs nested cross-validation with 
    5 inner and 10 outer folds training Logistic Regression and GBRT models 
    for each of the feature sets described in Table 1.
5. `modeling/prefix_results.ipynb`: takes the results of the previous step and 
    generates a CSV in a format convenient for plotting. 
6. `plotting/prefix_results.R`: generates Fig 7.
7. `modeling/prefix_domain_transfer.py`: performs the domain transfer 
    experiments (i.e., training on one dataset and testing on another) 
    described in Sec 5.3.


### Next reply toxicity predictions (Sec 5.2)
Run the following scripts:
1. `modeling/next_reply_sampling.ipynb`: samples pairs of toxic and nontoxic 
    replies for the paired prediction task.
2. `processing/next_reply_metrics.py`: compute the metrics described in Table 2.
3. `modeling/next_reply_make_datasets.ipynb`: builds the design matrix (X) and
    the labels vector (y).
4. `modeling/next_reply_prediction.py`: performs nested cross-validation with 
    5 inner and 10 outer folds training Logistic Regression and GBRT models 
    for each of the feature sets described in Table 2.
5. `modeling/next_reply_results.ipynb`: takes the results of the previous step 
    and generates a CSV in a format convenient for plotting. 
6. `plotting/next_reply_results.R`: generates Fig 8.
7. `modeling/next_reply_domain_transfer.ipynb`: performs the domain transfer 
    experiments (i.e., training on one dataset and testing on another) 
    described in Sec 5.3.



## Data guide
We collected two datasets, one of conversations started by tweets that are 
posted by or mention five major news outlets (news dataset), and another one of 
conversations prompted by tweets that are posted by or mention 1,430 politicians 
that ran for office during the 2018 midterm elections in the US 
(midterms dataset). 

The anonymized versions of the datasets are available at the 
[data repository hosted on the Harvard Dataverse](https://doi.org/10.7910/DVN/TRWPVB).

### Data organization
Since the embeddedness (i.e., number of common friends) statistics among the 
conversation participants require a lot of storage, we provide two versions 
of the datasets: one with (`news_jsons.7z.xxx` and `midterms_jsons.7z.xxx`) and 
another without (`news_jsons_no_embs.7z.001` and `midterms_jsons_no_embs.7z.001`)
the embeddedness information.

Size of the compressed data:
- `news_jsons.7z`: 7zip archive of the full news dataset split into 15 files
     (max 2GB each), total of 29.5GB.
- `midterms_jsons.7z`: 7zip archive of the full midterms dataset split into 
    38 files (max 2GB each), total of 74.2GB.
- `news_jsons_no_embs.7z`: 7zip archive of the news dataset excluding 
    the embeddedness information, one file of 1.2GB.
- `midterms_jsons_no_embs.7z`: 7zip archive of the midterms dataset excluding 
    the embeddedness information, one file of 1.2GB.

Size of the corresponding datasets uncompressed:
- `jsons/news`: 32GB
- `jsons/midterms`: 77GB
- `jsons_no_embs/news`: 2.8GB
- `jsons_no_embs/midterms`: 3.6GB

To uncompress the data run the following [7zip](https://www.7-zip.org/) commands:
```
7za x news_jsons.7z
7za x midterms_jsons.7z
7za x news_jsons_no_embs.7z
7za x midterms_jsons_no_embs.7z
```

### Data structure
Each dataset version is stored in a separate directory in which there is one 
json (gzipped) file per conversation. 

The json files have the following structure:

```
{
    "root_tweet_type": "post|mention",
    "tweets": {
        "tweet_id_0": { 
            "id": "tweet_id_0",
            "user_id": "user_id_0",
            "time": 0
        },
        "tweet_id_1": { 
            "id": "tweet_id_1",
            "user_id": "user_id_1",
            "time": 30
        },
        ...
    },
    "reply_tree": {
        "tweet": "tweet_id_0",
        "replies": [
            {
                "tweet": "tweet_id_1",
                "replies": []
            },
            {
                "tweet": "tweet_id_2",
                "replies": [
                    {
                        "tweet": "tweet_id_3",
                        "replies": []
                    }
                ]
            },
            ...
        ]
    },
    "network_features": {
        "user_ids": [
            "user_id_0",
            "user_id_1",
            ...
        ],
        "missing_user_ids": [
            "user_id_25"
        ],
        "network": [
            [0, 1],
            [3, 1],
            ...
        ],
        "network_intersections": [
            [0, 2, 1],
            [0, 3, 3],
            ...
        ],            
        "n_friends": [
            1111,
            55,
            ...
        ],
        "n_followers": [
            15838282,
            1040,
            ...
        ]
    }
    "alignment_scores": {
        "user_id_0": 0.2717478138,
        "user_id_1": -0.1746591394,
        ...
    },
    "toxicity_scores": {
        "tweet_id_0": 0.06521354,
        "tweet_id_1": 0.059626743,
        ...
    }
}
```

Description of fields:
- `root_tweet_type`: whether the root tweet was a post or a mention of the 
    accounts we tracked,
- `tweets`: dictionary of dictionaries, one per tweet: tweet_id (str) => 
    tweet info,
- `reply_tree`: tree-like data structure, each "node" is a dictionary and 
    contains the tweet_id (str) and the list of replies to it, which in turn 
    are also dictionaries with the same structure.
- `network_features`: contains the follow graph information associated with 
    the conversation participants
    - `user_ids`: list of user ids, all subsequent fields refer to the users 
        by indexing this list.
    - `missing_user_ids`: user ids for which we could not obtain the follow 
        graph information, hence are not included in the subsequent fields.
    - `network`: i, j pair where user with id `user_ids[i]` follow user with 
        id `user_ids[j]`.
    - `network_intersections`: number of common friends between each pair of 
        nodes. Triple `[i, j, k]` means that `user_ids[i]` and `user_ids[j]` 
        have `k` friends in common. If a pair is missing, the two users do not 
        have any friends in common. (NB: Friends here refers to outgoing links 
        in the Twitter follow graph.)
    - `n_friends`: number of friends (outgoing edges) the user has in the 
        full Twitter graph. Follows the order in `user_ids`.
    - `n_followers`: number of followers (incoming edges) the user has in the 
        full Twitter graph. Follows the order in `user_ids`.
- `alignment_scores`: political alignment scores of the users ranging between 
    -1 (left-leaning) and +1 (right-leaning) computed based on the users'
    content sharing patterns. 
- `toxicity_scores`: toxicity score of each tweet as predicted by Google's
    Perspective API. Refer to the documentation of the Perspective API for 
    proper interpretation of these scores. We used the most recent version of 
    the Toxicity model (TOXICITY@6) at the time, released in Sep 2018.

Note that the difference between the full datasets and the versions of the 
datasets that do not include the embeddedness information is that the field 
`network_features` -> `network_intersections` is missing in the latter.

### Anonymization
Although all the information we used to build the dataset is public (public 
tweets and user profiles), we decided to take extra steps and anonymize the 
data to protect the users' privacy. We believe that this is especially 
important in this study, where we classify  (sometimes inaccurately) 
some of the users' tweets as toxic.

We took the following steps to anonymize the data:
- Replaced all tweet and user ids, 
- Removed the tweet text,
- Normalized the tweet creation time relative to when the root tweet was 
    posted (i.e., tweet_time - root_tweet_time in seconds).

Note that since we anonymized the user ids, it is impossible to track users 
across different conversations in the dataset.

### Derived Data
The data repository also contains intermediary data derived for the purposes of 
the analyses (`derived_data.7z`). We decided to share this data as many of the 
aggregate statistics included in these files take a lot of time and memory to 
compute due to the size of the datasets.

The files include the outputs of various conversation metrics on the full 
conversations (Sec 4) and the outputs of the metrics used as inputs for the 
future toxicity and next reply prediction tasks (Sec 5). This data can be 
particularly useful if one is interested only in certain aggregate statistics 
on the conversations. 

The complete list of output files can be found in `DERIVED_DATA_FILES.md`. The 
files in the `news` and `midterms` directories are generated using the 
similarly named scripts in `code/processing` ran on the complete conversations. 
The files in the `modeling` directory can be mapped to the notebooks and 
scripts in code/modeling.


## Citation
```
@inproceedings{saveski2021structure,
    title={The Structure of Toxic Conversations on Twitter},
    author={Saveski, Martin and Roy, Brandon and Roy, Deb},
    year={2021}
    publisher = {Association for Computing Machinery},
    booktitle = {Proceedings of The Web Conference 2021},
    series = {WWW '21}
}
```


## License
This code is licensed under the MIT license found in the LICENSE file.
