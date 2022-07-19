from train import train_model



if __name__ == '__main__':
    # The dataset can be changed into either 'BRIAREO' or 'FPHA'
    dataset_name="SHREC17"
    train_model(dataset_name)