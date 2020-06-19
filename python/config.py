class Config:
    
    tmp_save_dir='../../models_python'
    train_num_workers=6
    test_num_workers=3
    
        
    # train_num_workers=0
    # test_num_workers=0
    

    # data_path='../../CT_rotation_data_npy_128'
    
    # model_name='Aug3D'
    
    # lvl1_size=4
    
    # is3d=True
    
    # train_batch_size = 8
    # test_batch_size = 4
    
        
    # max_epochs = 14
    # step_size=6
    # gamma=0.1
    # init_lr=0.001
    
    
    



    data_path='../../CT_rotation_data_2D'
    
    model_name='NoAug2D'
    
    is3d=False
    
    train_batch_size = 32
    test_batch_size = 32
    
        
    max_epochs = 12
    step_size=5
    gamma=0.1
    init_lr=0.001
    

    pretrained=True
    
    
    
    
    