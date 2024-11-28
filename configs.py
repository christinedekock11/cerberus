# shapes of input matrices 
torch_config = {
    'C_path': 'data/C*.mtx.gz',
    'A_path': 'data/A*.mtx.gz',
    'lr': 0.001, 'device':'cpu',
    'num_epochs': 100,
    'K':100, 'weight_A':1, 'weight_C':1, 'c0_scaler':0.01,
    'lambda_1':1, 'lambda_2':1, 'batch_size':80,
    'early_stopping_p':5, 'early_stopping_tol':0.001
}

pl_config = {
    'C_path': 'data/C*.mtx.gz',
    'A_path': 'data/A*.mtx.gz',
    'lr': 0.001, 'accelerator':'cpu', 'devices':1,#'devices':[0],'accelerator':'gpu',
    'num_epochs': 100,
    'K':100, 'weight_A':1, 'weight_C':1, 'c0_scaler':0.01,
    'lambda_1':1, 'lambda_2':1, 'batch_size':80,
    'early_stopping_p':5, 'early_stopping_tol':0.001
}
