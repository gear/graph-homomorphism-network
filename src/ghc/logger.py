import os
import pandas as pd
import time

class Logger(object):
    """Util class to log and write experiment results."""
    def __init__(self, args, path="./log"):
        self.args = args.__dict__
        self.id = str(time.time()).split('.')[0]
        self.log_file = os.path.join(path, args.dataset) 
        os.makedirs(self.log_file, exist_ok=True)
        self.log_file = os.path.join(self.log_file, self.id+".csv")
        with open(self.log_file, "w") as f:
            for k, v in self.args.items():
                f.write("# {}: {}\n".format(k, v))

    def write_log(self, log_dict, fold_id): 
        fold_data = [fold_id] * len(log_dict["epoch"])
        log_dict["fold"] = fold_data
        data = pd.DataFrame.from_dict(log_dict)
        header = True if fold_id == 1 else False
        data.to_csv(self.log_file, index=False, mode="a", header=header)
        
