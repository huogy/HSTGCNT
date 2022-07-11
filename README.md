# HSTGCNT 

# Dataset

Traffic data files for the Bay Area (PeMS-BAY) are available at https://github.com/liyaguang/DCRNN.

Traffic data files for the D7 Area (PeMS-BAY) are available at https://github.com/VeritasYin/STGCN_IJCAI-18.

If you need the Beijing Metro dataset, you need to contact the author. Email: gyhuo@emails.bjut.edu.cn


# Envirnoment Set-Up 

Clone the git project:

```
$ git clone https://github.com/huogy/HSTGCNT
```

Create a new conda Environment and install required packages (Commands are for ubuntu 16.04)

```
$ conda create -n HSTGCNTEnv python=3.7
$ conda activate HSTGCNTEnv
$ pip install -r requirements.txt
```

# Basic Usage:

**Main Parameters:**

```
matrix_path         Adjacency matrix
data_path           The input traffic flow data
save_path           Path to store the model
save_path_AE        Path to store the LTT
n_his               Input data sequence length
n_pred              Predict data sequence length
n_route             Number of road segments of the data
batch_size          Batch_size
epochs              Number of epochs during training
lr                  Learning rate
```

**Pre-Train LTT Using:**

```
$ python run_LTT.py
```

**Train Model Using:**

```
$ python main_HSTGCNT.py
```