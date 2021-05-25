


class Config():

    def __init__(self, dataset=None):
        
        assert dataset in (None, "news", "midterms")

        self.root = "/path/to/the/code"
        self.jsons_root_dir = "/path/to/the/raw/data"

        if dataset == "news":
            self.__set_news_paths__()

        elif dataset == "midterms":
            self.__set_midterms_paths__()

        self.__set_common_paths__()


    def __set_news_paths__(self):

        self.data_root = f"{self.root}/data/news"

        self.conversations_jsons_dir = f"{self.jsons_root_dir}/jsons/news/"

        self.conversations_no_embs_jsons_dir = (f"{self.jsons_root_dir}"
            "/jsons_no_embs/news/")


    def __set_midterms_paths__(self):

        self.data_root = f"{self.root}/data/midterms"

        self.conversations_jsons_dir = f"{self.jsons_root_dir}/jsons/midterms/"

        self.conversations_no_embs_jsons_dir = (f"{self.jsons_root_dir}"
            "/jsons_no_embs/midterms/")


    def __set_common_paths__(self):

        # modeling path
        self.modeling_dir = f"{self.root}/data/modeling"

# END
